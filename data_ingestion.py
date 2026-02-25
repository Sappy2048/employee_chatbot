# ingestion.py
import pandas as pd
import re
import json
import sqlite3
from datetime import datetime, timezone
from pydantic import BaseModel
from typing import List, Dict


# ── STEP 1: LOAD & PARSE ──────────────────────────────────────────────────────

def parse_shap_string(shap_str: str) -> Dict[str, float]:
    matches = re.findall(r'([\w_]+)\(([\d.]+)\)', str(shap_str))
    return {feat: float(score) for feat, score in matches}

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["is_selected"] = df["is_selected"].apply(
        lambda x: x if isinstance(x, bool) else str(x).strip().upper() == "TRUE"
    )
    df["shap_parsed"] = df["shap_values"].apply(parse_shap_string)
    return df


# ── STEP 2: BASELINE FROM ENGAGED EMPLOYEES ───────────────────────────────────

def compute_baseline(df: pd.DataFrame) -> Dict[str, float]:
    false_df = df[~df["is_selected"]]
    all_features = set(feat for d in false_df["shap_parsed"] for feat in d)
    return {
        feat: round(
            sum(row.get(feat, 0.0) for row in false_df["shap_parsed"]) / len(false_df), 6
        )
        for feat in all_features
    }


# ── STEP 3: PYDANTIC PROFILE SCHEMA ───────────────────────────────────────────

class EmployeeProfile(BaseModel):
    employee_id: str
    is_selected: bool
    top_issues: List[str]
    shap_scores: Dict[str, float]
    severity_scores: Dict[str, float]
    zero_shap: bool

def get_top_issues(shap_dict: Dict[str, float], top_n: int = 3) -> List[str]:
    if not shap_dict or all(v == 0.0 for v in shap_dict.values()):
        return ["general_disengagement"]
    return sorted(shap_dict, key=shap_dict.get, reverse=True)[:top_n]

def compute_severity(shap_dict, baseline) -> Dict[str, float]:
    return {
        feat: round(score - baseline.get(feat, 0.0), 6)
        for feat, score in shap_dict.items()
    }

def build_profiles(df: pd.DataFrame, baseline: Dict[str, float]) -> List[EmployeeProfile]:
    profiles = []
    for _, row in df[df["is_selected"]].iterrows():
        shap_dict = row["shap_parsed"]
        profiles.append(EmployeeProfile(
            employee_id=row["employee_id"],
            is_selected=True,
            top_issues=get_top_issues(shap_dict),
            shap_scores=shap_dict,
            severity_scores=compute_severity(shap_dict, baseline),
            zero_shap=all(v == 0.0 for v in shap_dict.values()),
        ))
    return profiles


# ── STEP 4: PERSIST ───────────────────────────────────────────────────────────

def persist(profiles: List[EmployeeProfile]):
    # 4a — JSON for LangGraph agent
    with open("profiles.json", "w") as f:
        json.dump([p.model_dump() for p in profiles], f, indent=2)

    # 4b — SQLite for scheduler, reports, email notifier
    conn = sqlite3.connect("engagement.db")
    conn.execute("DROP TABLE IF EXISTS employee_profiles")
    conn.execute("""
        CREATE TABLE employee_profiles (
            employee_id     TEXT PRIMARY KEY,
            top_issues      TEXT,
            shap_scores     TEXT,
            severity_scores TEXT,
            zero_shap       INTEGER,
            ingested_at     TEXT
        )
    """)
    now = datetime.now(timezone.utc).isoformat()
    for p in profiles:
        conn.execute("INSERT INTO employee_profiles VALUES (?,?,?,?,?,?)", (
            p.employee_id,
            json.dumps(p.top_issues),
            json.dumps(p.shap_scores),
            json.dumps(p.severity_scores),
            int(p.zero_shap),
            now,
        ))
    conn.commit()
    conn.close()


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def run_ingestion(filepath: str = "engagement_results.csv"):
    df       = load_data(filepath)
    baseline = compute_baseline(df)
    profiles = build_profiles(df, baseline)
    persist(profiles)
    print(f"[Ingestion] {len(profiles)} profiles written | "
          f"Zero-SHAP: {sum(p.zero_shap for p in profiles)}")
    return profiles

if __name__ == "__main__":
    run_ingestion()
