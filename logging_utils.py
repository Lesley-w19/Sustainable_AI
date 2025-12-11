# logging_utils.py
import os
import csv
import json
import datetime as dt


def log_energy_event(
    action: str,
    variant: str,
    prompt_text: str,
    features: dict,
    energy_kwh: float,
    layers: int,
    training_hours: float,
    flops_hr: float,
):
    """
    Log each prompt + energy estimate to both CSV and JSONL.

    This simulates how a provider could start building
    an energy reporting pipeline for regulatory reporting.
    """
    timestamp = dt.datetime.utcnow().isoformat() + "Z"

    row = {
        "timestamp_utc": timestamp,
        "action": action,               # "check" or "improve"
        "variant": variant,             # "original" or "improved"
        "prompt_preview": prompt_text[:200].replace("\n", " "),
        "energy_kwh": energy_kwh,
        "layers": layers,
        "training_hours": training_hours,
        "flops_hr": flops_hr,
        "tokens": features.get("tokens", 0.0),
        "type_token_ratio": features.get("type_token_ratio", 0.0),
        "avg_sentence_len": features.get("avg_sentence_len", 0.0),
        "stopword_ratio": features.get("stopword_ratio", 0.0),
        "sections": features.get("sections", 0.0),
    }

    # ---- CSV logging ----
    csv_path = "energy_logs.csv"
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    # ---- JSONL logging ----
    jsonl_path = "energy_logs.jsonl"
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
