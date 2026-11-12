"""
Train iris disease classifier with augmentation and two-phase fine-tuning.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from tensorflow.keras.utils import to_categorical

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    BATCH_SIZE,
    DEFAULT_MODEL_PROFILE,
    EPOCHS,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LEARNING_RATE,
    IMG_SIZE,
    LEARNING_RATE,
    MODEL_PROFILES,
    class_names_path,
    model_meta_path,
    training_metrics_path,
    weights_path,
)
from src.models.build_models import compile_model, get_model, unfreeze_top_layers  # noqa: E402

AUTOTUNE = tf.data.AUTOTUNE


def _group_id(img_path: Path) -> str:
    """
    Group id for leakage-safe splitting: augmented exports (e.g. Roboflow)
    share a leading numeric source-image id across derived copies
    ('10_JPG.rf.<hash>.jpg', '10_JPG_jpg.rf.<hash>.jpg' both come from
    source photo 10); files without that pattern are their own group.
    Prevents augmented siblings of one source photo — or same-patient
    left/right images sharing an id prefix — from splitting across
    train and validation.
    """
    prefix = img_path.stem.split("_", 1)[0]
    return prefix if prefix.isdigit() else img_path.stem


def load_dataset(data_path: Path):
    images, labels, groups, class_names = [], [], [], sorted(
        [d.name for d in data_path.iterdir() if d.is_dir()]
    )
    if not class_names:
        raise FileNotFoundError(f"No class folders found in {data_path}.")

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for label_idx, class_name in enumerate(class_names):
        for img_path in (data_path / class_name).iterdir():
            if img_path.suffix.lower() in exts:
                raw = tf.io.read_file(str(img_path))
                img = tf.image.decode_image(raw, channels=3, expand_animations=False)
                img = tf.image.resize(img, IMG_SIZE)
                img = tf.cast(img, tf.float32) / 255.0
                images.append(img.numpy())
                labels.append(label_idx)
                groups.append(_group_id(img_path))

    if not images:
        raise FileNotFoundError(f"No images found under {data_path}")

    x = np.array(images, dtype=np.float32)
    y = to_categorical(np.array(labels), num_classes=len(class_names))
    return x, y, np.array(groups), class_names


def make_augmentation_layer() -> tf.keras.Sequential:
    """Gentle augmentation suited for medical eye images."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def build_tf_dataset(x, y, batch_size: int, training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        augment = make_augmentation_layer()

        def _augment(img, label):
            img = augment(img, training=True)
            img = tf.clip_by_value(img, 0.0, 1.0)
            return img, label

        ds = ds.shuffle(min(len(x), 2000), seed=42).map(_augment, num_parallel_calls=AUTOTUNE)

    return ds.batch(batch_size).prefetch(AUTOTUNE)


def evaluate_model(model, x_val, y_val, class_names: list[str]) -> dict:
    y_true = np.argmax(y_val, axis=1)
    probs = model.predict(x_val, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    return {
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "accuracy": report["accuracy"],
    }


def main():
    parser = argparse.ArgumentParser(description="Train iris disease classifier")
    parser.add_argument("--profile", choices=list(MODEL_PROFILES), default=DEFAULT_MODEL_PROFILE)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--model", choices=["cnn", "mobilenet", "resnet50"], default="mobilenet")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--fine-tune-epochs", type=int, default=FINE_TUNE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--no-fine-tune", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    data_dir = args.data or MODEL_PROFILES[args.profile]["default_data_dir"]
    model_dir = MODEL_PROFILES[args.profile]["dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = training_metrics_path(args.profile)
    best_weights = weights_path(args.profile)
    args.data = data_dir

    print(f"Loading dataset from {args.data}...")
    x, y, groups, class_names = load_dataset(args.data)
    print(f"Loaded {len(x)} images ({len(set(groups))} unique source photos) | classes: {class_names}")

    label_idx_all = np.argmax(y, axis=1)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(splitter.split(x, label_idx_all, groups=groups))
    x_train, x_val = x[train_idx], x[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    print(
        f"Train/val split: {len(x_train)}/{len(x_val)} images, "
        f"{len(set(groups[train_idx]))}/{len(set(groups[val_idx]))} unique source photos "
        "(grouped so no source photo's augmented copies span both sides)"
    )

    # Balance loss for classes with fewer correct predictions (e.g. glaucoma)
    train_labels = np.argmax(y_train, axis=1)
    class_counts = np.bincount(train_labels, minlength=len(class_names))
    class_weights = {
        i: float(len(train_labels)) / (len(class_names) * max(count, 1))
        for i, count in enumerate(class_counts)
    }
    print(f"Class weights: {class_weights}")

    if args.no_augment:
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(2000).batch(args.batch_size).prefetch(AUTOTUNE)
    else:
        train_ds = build_tf_dataset(x_train, y_train, args.batch_size, training=True)
    val_data = (x_val, y_val)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(best_weights),
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        mode="max",
        verbose=1,
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True, mode="max"
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1
    )

    model = get_model(args.model, num_classes=len(class_names), trainable_base=False)
    print(f"\nPhase 1: training {args.model} head (max {args.epochs} epochs)...")
    history1 = model.fit(
        train_ds,
        validation_data=val_data,
        epochs=args.epochs,
        callbacks=[checkpoint, early_stop, reduce_lr],
        class_weight=class_weights,
    )
    phase1_best = max(history1.history["val_accuracy"])
    print(f"Phase 1 best val accuracy: {phase1_best:.2%}")

    if not args.no_fine_tune and args.model in ("mobilenet", "resnet50"):
        model.load_weights(str(best_weights))
        model = unfreeze_top_layers(model, num_layers=30)
        model = compile_model(model, learning_rate=FINE_TUNE_LEARNING_RATE)
        print(f"\nPhase 2: fine-tuning top layers (max {args.fine_tune_epochs} epochs)...")
        history2 = model.fit(
            train_ds,
            validation_data=val_data,
            epochs=args.fine_tune_epochs,
            callbacks=[checkpoint, early_stop, reduce_lr],
            class_weight=class_weights,
        )
        phase2_best = max(history2.history["val_accuracy"])
        print(f"Phase 2 best val accuracy: {phase2_best:.2%}")

    model.load_weights(str(best_weights))

    metrics = evaluate_model(model, x_val, y_val, class_names)
    print(f"\nFinal validation accuracy: {metrics['accuracy']:.2%}")
    print("\nPer-class report:")
    for name in class_names:
        stats = metrics["classification_report"][name]
        print(f"  {name}: P={stats['precision']:.2f} R={stats['recall']:.2f} F1={stats['f1-score']:.2f}")

    meta_path = model_meta_path(args.profile)
    meta_path.write_text(
        json.dumps({"model_type": args.model, "num_classes": len(class_names)}, indent=2)
    )
    class_names_path(args.profile).write_text("\n".join(class_names))
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nWeights saved to {best_weights}")
    print(f"Model meta saved to {meta_path}")


if __name__ == "__main__":
    main()
