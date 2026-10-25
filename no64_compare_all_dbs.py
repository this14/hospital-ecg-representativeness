#!/usr/bin/env python3
"""No.64: 병원 안 DB(EchoNext, Georgia, SNUH, PTB-XL, MIMIC) vs 웨어러블 사용 인구(HINTS 2024) 비교표.

MIMIC은 mimic_table1_in_hospital.py 결과(table1_first_ecg_per_patient.csv)를 읽는다.
나머지 DB는 여기서 직접 집계한다. 출력은 집계 CSV뿐이다(개인 단위 행 저장 안 함).

사용법:
    python3 no64_compare_all_dbs.py --out /home/this14/no64_out

기본 경로(서버): 필요하면 --echonext/--georgia/--snuh/--ptbxl/--mimic_table1 로 바꾼다.

사전 지정 규칙:
  - 성인(18세 이상)만. 90세 이상은 90으로 묶는 DB가 많아 구간 비교에만 사용(연속 나이 통계 제외)
  - 환자 ID가 있으면 환자당 1건(가장 이른 기록): EchoNext는 acquisition_year가 가장 이른 행(동률은 파일 순서),
    PTB-XL은 recording_date가 가장 이른 행. Georgia와 SNUH는 환자 ID가 없거나 불확실해 ECG(행) 단위
  - PTB-XL 나이 300은 '89세 초과' 익명화 코드이므로 90으로 처리
  - SNUH 나이는 18~110세만 인정(0세, 109세 초과 등은 입력 오류 가능성으로 제외하고 건수를 기록)
  - 총변동거리(TV) = 0.5 * sum |p_DB - p_HINTS| (단위: %p). 인종은 '미상'을 제외하고 White/Black/Hispanic/Asian/Other 5범주로 비교
  - HINTS 기준값은 hints_reference.py로 재계산한 2024 웨어러블 사용자 값(나이·성별·인종·BMI)과 일치한다
"""
import argparse
import glob
import os
import re
import numpy as np
import pandas as pd

AGE_BINS = [18, 35, 50, 65, 75, 200]
AGE_LABELS = ["18-34", "35-49", "50-64", "65-74", "75+"]
BMI_BINS = [0, 18.5, 25, 30, 1000]
BMI_LABELS = ["<18.5", "18.5-24.9", "25-29.9", ">=30"]
RACE_CATS = ["White", "Black", "Hispanic", "Asian", "Other"]

# HINTS 2024 웨어러블 사용자 (%) - 이전 세션 계산값(가중, 재계산 전)
HINTS_AGE = dict(zip(AGE_LABELS, [29.7, 31.1, 26.7, 8.4, 4.1]))
HINTS_RACE = dict(zip(RACE_CATS, [61.4, 10.9, 17.7, 5.1, 4.9]))
HINTS_FEMALE = 52.2
# HINTS 2024 웨어러블 사용자 BMI 분포(%, 재계산 완료: hints_reference.py)
HINTS_BMI = dict(zip(BMI_LABELS, [1.2, 28.0, 33.8, 37.0]))


def summarize(df, db, group):
    rows = []
    n = len(df)
    base = {"db": db, "group": group, "n": n}
    age_cat = pd.cut(df["age"], AGE_BINS, labels=AGE_LABELS, right=False)
    age_pct = (age_cat.value_counts(normalize=True).reindex(AGE_LABELS).fillna(0) * 100)
    for k in AGE_LABELS:
        rows.append({**base, "variable": "age", "category": k, "pct": age_pct[k]})
    rows.append({**base, "variable": "female", "category": "", "pct": 100 * (df["female"] == 1).mean()})
    rows.append({**base, "variable": "age18_49", "category": "", "pct": age_pct[["18-34", "35-49"]].sum()})
    rows.append({**base, "variable": "age65plus", "category": "", "pct": age_pct[["65-74", "75+"]].sum()})
    rows.append({**base, "variable": "TV_age_vs_HINTS", "category": "",
                 "pct": 0.5 * sum(abs(age_pct[k] - HINTS_AGE[k]) for k in AGE_LABELS)})
    rows.append({**base, "variable": "female_diff_vs_HINTS", "category": "",
                 "pct": 100 * (df["female"] == 1).mean() - HINTS_FEMALE})
    if "race5" in df.columns:
        race = df["race5"].fillna("Unknown")
        rows.append({**base, "variable": "race_unknown", "category": "", "pct": 100 * (race == "Unknown").mean()})
        known = race[race != "Unknown"]
        if len(known):
            rp = (known.value_counts(normalize=True).reindex(RACE_CATS).fillna(0) * 100)
            for k in RACE_CATS:
                rows.append({**base, "variable": "race_excl_unknown", "category": k, "pct": rp[k]})
            rows.append({**base, "variable": "TV_race_vs_HINTS", "category": "",
                         "pct": 0.5 * sum(abs(rp[k] - HINTS_RACE[k]) for k in RACE_CATS)})
    if "bmi" in df.columns:
        has = df["bmi"].notna()
        rows.append({**base, "variable": "bmi_available", "category": "", "pct": 100 * has.mean()})
        if has.any():
            bc = pd.cut(df.loc[has, "bmi"], BMI_BINS, labels=BMI_LABELS, right=False)
            bp = bc.value_counts(normalize=True).reindex(BMI_LABELS).fillna(0) * 100
            for k in BMI_LABELS:
                rows.append({**base, "variable": "bmi_among_available", "category": k, "pct": bp[k]})
            rows.append({**base, "variable": "bmi_median", "category": "", "pct": df.loc[has, "bmi"].median()})
            rows.append({**base, "variable": "TV_bmi_vs_HINTS", "category": "",
                         "pct": 0.5 * sum(abs(bp[k] - HINTS_BMI[k]) for k in BMI_LABELS)})
    return rows


def echonext(path, notes):
    e = pd.read_csv(path, usecols=["patient_key", "age_at_ecg", "sex", "acquisition_year",
                                   "location_setting", "race_ethnicity"])
    notes.append(f"EchoNext: {len(e)} ECG rows, {e['patient_key'].nunique()} patients")
    e = e[e["age_at_ecg"] >= 18]
    e = e.reset_index().sort_values(["acquisition_year", "index"]).drop_duplicates("patient_key")
    e["age"] = e["age_at_ecg"]
    e["female"] = (e["sex"].str.lower() == "female").astype(int)
    m = {"white": "White", "black": "Black", "hispanic": "Hispanic", "asian": "Asian", "other": "Other"}
    e["race5"] = e["race_ethnicity"].str.lower().map(m).fillna("Unknown")
    e["setting"] = e["location_setting"].str.lower()
    return e


def georgia(root, notes):
    recs = []
    for f in glob.glob(os.path.join(root, "**", "*.hea"), recursive=True):
        age = sex = None
        with open(f, "r", errors="replace") as fh:
            for ln in fh:
                if ln.startswith("# Age:"):
                    age = ln.split(":", 1)[1].strip()
                elif ln.startswith("# Sex:"):
                    sex = ln.split(":", 1)[1].strip()
        recs.append((age, sex))
    g = pd.DataFrame(recs, columns=["age", "sex"])
    n0 = len(g)
    g["age"] = pd.to_numeric(g["age"], errors="coerce")
    notes.append(f"Georgia: {n0} headers, age missing/NaN {int(g['age'].isna().sum())}")
    g = g[g["age"] >= 18].copy()
    g["female"] = (g["sex"].str.lower() == "female").astype(int)
    return g


def snuh(path, notes):
    s = pd.read_csv(path, usecols=["new_id", "age", "sex"])
    n0 = len(s)
    notes.append(f"SNUH: {n0} rows, {s['new_id'].nunique()} unique ids, age missing {int(s['age'].isna().sum())}")
    bad = ((s["age"] < 18) | (s["age"] > 110)).sum()
    notes.append(f"SNUH: excluded age<18 or >110: {int(bad)}")
    s = s[(s["age"] >= 18) & (s["age"] <= 110)].copy()
    s["female"] = (s["sex"] == "F").astype(int)
    return s


def ptbxl(path, notes):
    x = pd.read_csv(path, usecols=["ecg_id", "patient_id", "age", "sex", "height", "weight", "recording_date"])
    notes.append(f"PTB-XL: {len(x)} ECG rows, {x['patient_id'].nunique()} patients")
    x["age"] = x["age"].where(x["age"] != 300, 90)
    x = x[x["age"] >= 18].copy()
    x["recording_date"] = pd.to_datetime(x["recording_date"], errors="coerce")
    x = x.sort_values("recording_date").drop_duplicates("patient_id")
    x["female"] = (x["sex"] == 1).astype(int)  # PTB-XL: 0=male, 1=female
    bmi = x["weight"] / (x["height"] / 100.0) ** 2
    x["bmi"] = bmi.where((bmi >= 10) & (bmi <= 80))
    return x


def mimic_rows(path):
    t = pd.read_csv(path)
    rows = []
    for grp, d in t.groupby("group"):
        base = {"db": "MIMIC-IV-ECG", "group": grp, "n": int(d.loc[d["variable"] == "N", "n_ecg"].iloc[0])}
        pick = d[d["variable"].isin(["age", "female", "race_excl_unknown", "race_incl_unknown",
                                     "bmi_available", "bmi_among_available", "bmi_median"])]
        a = pick[pick["variable"] == "age"].set_index("category")["pct"].reindex(AGE_LABELS).fillna(0)
        rows += [{**base, "variable": "age", "category": k, "pct": a[k]} for k in AGE_LABELS]
        f = pick[pick["variable"] == "female"]["pct"].iloc[0]
        rows.append({**base, "variable": "female", "category": "", "pct": f})
        rows.append({**base, "variable": "age18_49", "category": "", "pct": a[["18-34", "35-49"]].sum()})
        rows.append({**base, "variable": "age65plus", "category": "", "pct": a[["65-74", "75+"]].sum()})
        rows.append({**base, "variable": "TV_age_vs_HINTS", "category": "",
                     "pct": 0.5 * sum(abs(a[k] - HINTS_AGE[k]) for k in AGE_LABELS)})
        rows.append({**base, "variable": "female_diff_vs_HINTS", "category": "", "pct": f - HINTS_FEMALE})
        unk = pick[(pick["variable"] == "race_incl_unknown") & (pick["category"] == "Unknown")]["pct"]
        if len(unk):
            rows.append({**base, "variable": "race_unknown", "category": "", "pct": unk.iloc[0]})
        r = pick[pick["variable"] == "race_excl_unknown"].set_index("category")["pct"].reindex(RACE_CATS).fillna(0)
        rows += [{**base, "variable": "race_excl_unknown", "category": k, "pct": r[k]} for k in RACE_CATS]
        rows.append({**base, "variable": "TV_race_vs_HINTS", "category": "",
                     "pct": 0.5 * sum(abs(r[k] - HINTS_RACE[k]) for k in RACE_CATS)})
        for v in ("bmi_available", "bmi_median"):
            s = pick[pick["variable"] == v]["pct"]
            if len(s):
                rows.append({**base, "variable": v, "category": "", "pct": s.iloc[0]})
        b = pick[pick["variable"] == "bmi_among_available"].set_index("category")["pct"]
        for k in BMI_LABELS:
            if k in b.index:
                rows.append({**base, "variable": "bmi_among_available", "category": k, "pct": b[k]})
        if set(BMI_LABELS) <= set(b.index):
            rows.append({**base, "variable": "TV_bmi_vs_HINTS", "category": "",
                         "pct": 0.5 * sum(abs(b[k] - HINTS_BMI[k]) for k in BMI_LABELS)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    h = "/home/this14"
    ap.add_argument("--out", required=True)
    ap.add_argument("--echonext", default=f"{h}/us_representativeness/data/EchoNext/echonext_metadata_100k_v1.1.1.csv")
    ap.add_argument("--georgia", default=f"{h}/us_representativeness/data/georgia")
    ap.add_argument("--snuh", default=f"{h}/LYDUS_SNUH_ECG_ANONYMIZED_50K_0512.csv")
    ap.add_argument("--ptbxl", default=f"{h}/data/ptb-xl/ptbxl_database.csv")
    ap.add_argument("--mimic_table1", default=f"{h}/no64_out/table1_first_ecg_per_patient.csv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    notes, rows = [], []

    if os.path.exists(a.mimic_table1):
        rows += mimic_rows(a.mimic_table1)
    else:
        notes.append("MIMIC table1 not found: " + a.mimic_table1)

    e = echonext(a.echonext, notes)
    rows += summarize(e, "EchoNext", "Overall")
    for s in ("emergency", "inpatient", "outpatient", "procedural"):
        sub = e[e["setting"] == s]
        if len(sub):
            rows += summarize(sub, "EchoNext", s)
    rows += summarize(georgia(a.georgia, notes), "Georgia", "Overall")
    rows += summarize(snuh(a.snuh, notes), "SNUH", "Overall")
    p = ptbxl(a.ptbxl, notes)
    rows += summarize(p, "PTB-XL", "Overall")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.out, "compare_all_dbs_long.csv"), index=False)

    # 사람이 읽기 쉬운 요약표
    key = ["age18_49", "age65plus", "TV_age_vs_HINTS", "female", "TV_race_vs_HINTS", "race_unknown", "bmi_available", "TV_bmi_vs_HINTS"]
    wide = (out[out["variable"].isin(key)]
            .pivot_table(index=["db", "group", "n"], columns="variable", values="pct", aggfunc="first")
            .reindex(columns=[c for c in key if c in set(out["variable"])]).round(1))
    wide.to_csv(os.path.join(a.out, "compare_all_dbs_summary.csv"))
    with open(os.path.join(a.out, "compare_notes.txt"), "w") as f:
        f.write("\n".join(notes) + "\n")
        f.write("HINTS reference = HINTS 2024 wearable users recomputed with hints_reference.py (replicate weights). TV unit = percentage points.\n")
    print(wide.to_string())
    print("\n".join(notes))


if __name__ == "__main__":
    main()
