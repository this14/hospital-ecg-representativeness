#!/usr/bin/env python3
"""No.64: MIMIC-IV-ECG 병원 안 ECG의 인구통계 집계표 (나이, 성별, 인종, BMI, 환경).

사용법 (서버에서):
    python3 mimic_table1_in_hospital.py \
        --mimiciv /home/this14/mimiciv \
        --record_list /home/this14/mimic-iv-ecg/record_list.csv \
        --out /home/this14/no64_out

출력: --out 폴더에 집계 CSV만 저장한다(환자·ECG 단위 행은 저장하지 않음).
    table1_all_ecgs.csv         : 모든 ECG 기준
    table1_first_ecg_per_patient.csv : 환자당 첫 ECG 기준 (HINTS처럼 사람 단위 비교용)
    coverage_notes.txt          : 결측률, 제외 건수, 사용한 규칙

사전 지정한 규칙 (결과를 보기 전에 고정):
  - 성인(18세 이상)만 포함. 나이 = anchor_age + (ECG 연도 - anchor_year). 89세 초과는 anchor_age 91로 묶여 있음
  - BMI: OMR의 'BMI (kg/m2)' 또는 'BMI'. 없으면 같은 날 Height(Inches)/Weight(Lbs)로 계산(703*lbs/in^2).
         10 <= BMI <= 80 만 인정. ECG 시각과 가장 가까운 측정을 쓰되 +-365일 이내만 허용(--bmi_window로 변경 가능)
  - 인종: 환자당 입원 기록 중 '알 수 없음' 계열이 아닌 가장 흔한 라벨을 채택 -> 5개 범주로 묶음(아래 매핑)
  - 환경(linkage-derived): ECG 시각이 ED 구간이면 ED, 입원 구간이면 Inpatient, 어느 입원에도 속하지 않으면 Outside admission
"""
import argparse
import os
import numpy as np
import pandas as pd

AGE_BINS = [18, 35, 50, 65, 75, 200]
AGE_LABELS = ["18-34", "35-49", "50-64", "65-74", "75+"]
BMI_BINS = [0, 18.5, 25, 30, 1000]
BMI_LABELS = ["<18.5", "18.5-24.9", "25-29.9", ">=30"]
UNKNOWN_TOKENS = ("UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED")


def harmonize_race(label):
    """MIMIC admissions.race -> White / Black / Hispanic / Asian / Other / Unknown.
    선택 사항(논문 방법에 명시): SOUTH AMERICAN은 Hispanic, PORTUGUESE는 Other로 둔다."""
    if not isinstance(label, str):
        return "Unknown"
    s = label.upper()
    if any(t in s for t in UNKNOWN_TOKENS):
        return "Unknown"
    if "HISPANIC" in s or "SOUTH AMERICAN" in s:
        return "Hispanic"
    if "BLACK" in s:
        return "Black"
    if "ASIAN" in s:
        return "Asian"
    if "WHITE" in s:
        return "White"
    return "Other"  # OTHER, PORTUGUESE, AMERICAN INDIAN, NHPI, MULTIPLE RACE


def load_race(admissions):
    adm = admissions[["subject_id", "race"]].copy()
    adm["race5"] = adm["race"].map(harmonize_race)
    known = adm[adm["race5"] != "Unknown"]
    mode = (known.groupby("subject_id")["race5"]
            .agg(lambda x: x.value_counts().idxmax()))
    out = pd.Series("Unknown", index=adm["subject_id"].unique(), name="race5")
    out.loc[mode.index] = mode
    return out.rename_axis("subject_id").reset_index()


def load_bmi(omr_path):
    omr = pd.read_csv(omr_path, usecols=["subject_id", "chartdate", "result_name", "result_value"],
                      dtype={"result_name": "string", "result_value": "string"})
    omr["chartdate"] = pd.to_datetime(omr["chartdate"], errors="coerce")
    direct = omr[omr["result_name"].isin(["BMI (kg/m2)", "BMI"])].copy()
    direct["bmi"] = pd.to_numeric(direct["result_value"], errors="coerce")
    direct = direct[["subject_id", "chartdate", "bmi"]]

    hw = omr[omr["result_name"].isin(["Height (Inches)", "Weight (Lbs)"])].copy()
    hw["v"] = pd.to_numeric(hw["result_value"], errors="coerce")
    piv = (hw.pivot_table(index=["subject_id", "chartdate"], columns="result_name",
                          values="v", aggfunc="first").reset_index())
    if {"Height (Inches)", "Weight (Lbs)"} <= set(piv.columns):
        piv["bmi"] = 703.0 * piv["Weight (Lbs)"] / (piv["Height (Inches)"] ** 2)
        computed = piv[["subject_id", "chartdate", "bmi"]]
    else:
        computed = pd.DataFrame(columns=["subject_id", "chartdate", "bmi"])

    bmi = pd.concat([direct, computed], ignore_index=True)
    bmi = bmi.dropna(subset=["chartdate", "bmi"])
    bmi = bmi[(bmi["bmi"] >= 10) & (bmi["bmi"] <= 80)]
    # 같은 날짜 중복은 직접 BMI(먼저 들어온 것)를 우선
    bmi = bmi.drop_duplicates(["subject_id", "chartdate"], keep="first")
    return bmi.sort_values("chartdate").reset_index(drop=True)


def attach_setting(ecg, admissions):
    adm = admissions[["subject_id", "hadm_id", "admittime", "dischtime", "edregtime", "edouttime"]].copy()
    for c in ("admittime", "dischtime", "edregtime", "edouttime"):
        adm[c] = pd.to_datetime(adm[c], errors="coerce")
    adm["start"] = adm[["admittime", "edregtime"]].min(axis=1)
    adm = adm.dropna(subset=["start", "dischtime"]).sort_values("start").reset_index(drop=True)
    e = ecg.sort_values("ecg_time").reset_index(drop=True)
    m = pd.merge_asof(e, adm[["subject_id", "start", "dischtime", "admittime", "edouttime"]],
                      left_on="ecg_time", right_on="start", by="subject_id", direction="backward")
    inside = m["dischtime"].notna() & (m["ecg_time"] <= m["dischtime"])
    in_ed = inside & m["edouttime"].notna() & (m["ecg_time"] <= m["edouttime"]) & (m["ecg_time"] < m["admittime"])
    m["setting"] = np.where(in_ed, "ED", np.where(inside, "Inpatient", "Outside admission"))
    return m.drop(columns=["start", "dischtime", "admittime", "edouttime"])


def summarize(df, label):
    rows = []
    n = len(df)
    base = {"group": label, "n_ecg": n, "n_patients": df["subject_id"].nunique()}
    rows.append({**base, "variable": "N", "category": "", "pct": np.nan})

    age_cat = pd.cut(df["age"], AGE_BINS, labels=AGE_LABELS, right=False)
    for k, v in age_cat.value_counts(normalize=True).reindex(AGE_LABELS).items():
        rows.append({**base, "variable": "age", "category": k, "pct": 100 * v})
    rows.append({**base, "variable": "age_median", "category": "", "pct": df["age"].median()})
    rows.append({**base, "variable": "female", "category": "", "pct": 100 * (df["gender"] == "F").mean()})

    race = df["race5"].fillna("Unknown")
    for k, v in race.value_counts(normalize=True).items():
        rows.append({**base, "variable": "race_incl_unknown", "category": k, "pct": 100 * v})
    known = race[race != "Unknown"]
    for k, v in known.value_counts(normalize=True).items():
        rows.append({**base, "variable": "race_excl_unknown", "category": k, "pct": 100 * v})

    has_bmi = df["bmi"].notna()
    rows.append({**base, "variable": "bmi_available", "category": "", "pct": 100 * has_bmi.mean()})
    if has_bmi.any():
        bmi_cat = pd.cut(df.loc[has_bmi, "bmi"], BMI_BINS, labels=BMI_LABELS, right=False)
        for k, v in bmi_cat.value_counts(normalize=True).reindex(BMI_LABELS).items():
            rows.append({**base, "variable": "bmi_among_available", "category": k, "pct": 100 * v})
        rows.append({**base, "variable": "bmi_median", "category": "", "pct": df.loc[has_bmi, "bmi"].median()})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimiciv", required=True, help="MIMIC-IV 루트 (hosp 폴더의 상위)")
    ap.add_argument("--record_list", required=True, help="MIMIC-IV-ECG record_list.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bmi_window", type=int, default=365, help="ECG와 BMI 측정 간 허용 일수")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    hosp = os.path.join(a.mimiciv, "hosp")

    patients = pd.read_csv(os.path.join(hosp, "patients.csv.gz"),
                           usecols=["subject_id", "gender", "anchor_age", "anchor_year"])
    admissions = pd.read_csv(os.path.join(hosp, "admissions.csv.gz"))
    ecg = pd.read_csv(a.record_list, usecols=["subject_id", "study_id", "ecg_time"])
    ecg["ecg_time"] = pd.to_datetime(ecg["ecg_time"], errors="coerce")
    n0 = len(ecg)
    ecg = ecg.dropna(subset=["ecg_time"])

    df = ecg.merge(patients, on="subject_id", how="inner")
    n_linked = len(df)
    df["age"] = df["anchor_age"] + (df["ecg_time"].dt.year - df["anchor_year"])
    df = df[df["age"] >= 18].copy()
    n_adult = len(df)

    df = df.merge(load_race(admissions), on="subject_id", how="left")
    df = attach_setting(df, admissions)

    bmi = load_bmi(os.path.join(hosp, "omr.csv.gz")).rename(columns={"chartdate": "bmi_date"})
    df = df.sort_values("ecg_time").reset_index(drop=True)
    df = pd.merge_asof(df, bmi, left_on="ecg_time", right_on="bmi_date", by="subject_id",
                       direction="nearest", tolerance=pd.Timedelta(days=a.bmi_window))

    settings = ["Overall", "ED", "Inpatient", "Outside admission"]
    for name, data in (("all_ecgs", df),
                       ("first_ecg_per_patient", df.sort_values("ecg_time").drop_duplicates("subject_id"))):
        rows = []
        for s in settings:
            sub = data if s == "Overall" else data[data["setting"] == s]
            if len(sub):
                rows.extend(summarize(sub, s))
        pd.DataFrame(rows).to_csv(os.path.join(a.out, f"table1_{name}.csv"), index=False)

    with open(os.path.join(a.out, "coverage_notes.txt"), "w") as f:
        f.write(f"record_list rows: {n0}\n")
        f.write(f"after ecg_time parse + patients link: {n_linked}\n")
        f.write(f"adults (age>=18): {n_adult}\n")
        f.write(f"BMI window: +-{a.bmi_window} days\n")
        f.write(f"ECGs with BMI: {int(df['bmi'].notna().sum())} ({100*df['bmi'].notna().mean():.1f}%)\n")
        f.write(f"age>=89 at ECG (top-coded): {int((df['age'] >= 89).sum())}\n")
        f.write(df["setting"].value_counts().to_string() + "\n")
    print("saved to", a.out)


if __name__ == "__main__":
    main()
