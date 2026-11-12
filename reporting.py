"""Create downloadable screening reports from a completed prediction."""

from datetime import datetime
from html import escape

from src.inference import format_display_name


def build_screening_report(result: dict, profile_label: str, source: str, user: dict) -> tuple[bytes, str]:
    """Return a self-contained HTML report and a timestamped filename."""
    created_at = datetime.now().astimezone().strftime("%d %b %Y, %I:%M %p %Z")
    rows = [
        ("Account", user["name"]),
        ("Email", user["email"]),
        ("Scan source", source),
        ("Photo type", profile_label),
        ("Generated", created_at),
    ]
    if result.get("class_name"):
        rows.extend(
            [
                ("Primary screening result", format_display_name(result["class_name"])),
                ("Model confidence", f"{result['confidence'] * 100:.1f}%"),
            ]
        )
    else:
        rows.append(("Primary screening result", "Model unavailable — no model prediction produced"))

    probabilities = "".join(
        f"<li>{escape(format_display_name(label))}: {value * 100:.1f}%</li>"
        for label, value in (result.get("all_probs") or {}).items()
    )
    systemic_items = ""
    for item in (result.get("systemic") or {}).get("conditions", []):
        systemic_items += (
            f"<li><strong>{escape(item['label'])}</strong>: {escape(item['status'])} "
            f"(risk indicator {item['risk_score'] * 100:.0f}%)</li>"
        )
    details = "".join(f"<tr><th>{escape(key)}</th><td>{escape(value)}</td></tr>" for key, value in rows)
    html = f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>Iris Screening Report</title>
    <style>body{{font-family:Arial,sans-serif;color:#172554;max-width:800px;margin:40px auto;line-height:1.55}}
    h1{{color:#0f766e}} h2{{margin-top:28px;color:#164e63}} table{{border-collapse:collapse;width:100%}}
    th,td{{border:1px solid #cbd5e1;padding:10px;text-align:left}}th{{background:#f0fdfa;width:36%}}
    .notice{{background:#fff7ed;border-left:4px solid #f97316;padding:14px}}</style></head><body>
    <h1>Iris Health Screening Report</h1><p>AI-assisted screening summary</p><table>{details}</table>
    <h2>Class probabilities</h2><ul>{probabilities or '<li>Not available</li>'}</ul>
    <h2>Systemic screening indicators</h2><ul>{systemic_items or '<li>Not applicable for this photo type or unavailable</li>'}</ul>
    <p class=\"notice\"><strong>Important:</strong> This is an academic/research screening result, not a medical diagnosis. Please consult a qualified healthcare professional for diagnosis or treatment.</p>
    </body></html>"""
    filename = f"iris-screening-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.html"
    return html.encode("utf-8"), filename
