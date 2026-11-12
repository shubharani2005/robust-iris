"""Deep learning model builders for iris disease classification."""

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, ResNet50


def build_cnn_model(num_classes: int, input_shape: tuple[int, int, int] = (224, 224, 3)) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.3),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="iris_cnn",
    )
    return model


def _classification_head(x, num_classes: int) -> layers.Layer:
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    return layers.Dense(num_classes, activation="softmax")(x)


def build_mobilenet_model(
    num_classes: int,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    trainable_base: bool = False,
) -> keras.Model:
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = trainable_base

    inputs = keras.Input(shape=input_shape)
    x = base(inputs, training=trainable_base)
    outputs = _classification_head(x, num_classes)
    return keras.Model(inputs, outputs, name="iris_mobilenetv2")


def build_resnet50_model(
    num_classes: int,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    trainable_base: bool = False,
) -> keras.Model:
    base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = trainable_base

    inputs = keras.Input(shape=input_shape)
    x = base(inputs, training=trainable_base)
    outputs = _classification_head(x, num_classes)
    return keras.Model(inputs, outputs, name="iris_resnet50")


def compile_model(model: keras.Model, learning_rate: float = 1e-4) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_model(
    model_type: str,
    num_classes: int,
    trainable_base: bool = False,
    learning_rate: float = 1e-4,
) -> keras.Model:
    builders = {
        "cnn": build_cnn_model,
        "mobilenet": build_mobilenet_model,
        "resnet50": build_resnet50_model,
    }
    if model_type not in builders:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(builders)}")

    if model_type == "cnn":
        model = builders[model_type](num_classes)
    else:
        model = builders[model_type](num_classes, trainable_base=trainable_base)
    return compile_model(model, learning_rate=learning_rate)


def unfreeze_top_layers(model: keras.Model, num_layers: int = 30) -> keras.Model:
    """Unfreeze top N layers of the embedded ImageNet backbone."""
    backbone = None
    for layer in model.layers:
        if isinstance(layer, keras.Model) and len(layer.layers) > 50:
            backbone = layer
            break

    if backbone is None:
        return model

    backbone.trainable = True
    for layer in backbone.layers[:-num_layers]:
        layer.trainable = False
    return model
