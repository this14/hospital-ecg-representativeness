#!/usr/bin/env python3
"""No.64: 병원 ECG DB vs HINTS 2024 웨어러블 사용자 격차의 환자 단위 부트스트랩 CI + 나이x성별 커버리지(RQ2).

no64_compare_all_dbs.py와 같은 폴더에 두고 실행한다(로더 함수 재사용, 규칙 동일).
사용법:
    python3 no64_bootstrap_ci.py --out /home/this14/no64_out [--B 2000] [--seed 20261024]
        [--hints_joint /home/this14/no64_out/hints_age_sex_joint.csv] [--min_cell 100]

규칙(결과를 보기 전에 고정):
  - 분석 단위는 no64_compare_all_dbs.py와 같다: EchoNext, PTB-XL은 환자당 1건(가장 이른 기록)이므로
    행 = 환자이고, 환자 단위 부트스트랩 = 행 재추출. Georgia, SNUH는 환자 ID가 없거나 불확실해 ECG(행) 단위이며,
    같은 환자의 반복 기록이 있으면 CI가 좁게 나올 수 있다(표에 명시할 것).
  - 각 부트스트랩 반복마다 같은 크기로 복원추출 후 지표를 다시 계산. 신뢰구간 = 2.5, 97.5 백분위(percentile).
  - 지표: TV_age(HINTS 대비, %p), 여성 비율과 HINTS 대비 차이(%p), 65세 이상 비율, TV_race(미상 제외, 5범주)
  - HINTS 기준값은 고정값으로 취급한다(HINTS 자체의 표본오차는 별도 표에서 jackknife CI로 보고). 따라서 이 CI는
    병원 DB 쪽 표본오차만 반영한다.
  - 커버리지(RQ2): 나이 5구간 x 성별 2 = 10셀. 셀별 n, 비율(%), n<min_cell 표시.
    --hints_joint가 있으면 셀별 DB 비율/HINTS 비율(비), 'HINTS 사용자 중 DB가 n>=min_cell로 덮는 비율'을 계산.
"""
import argparse
import os
import numpy as np
import pandas as pd
import no64_compare_all_dbs as C

AGE_LABELS, RACE_CATS = C.AGE_LABELS, C.RACE_CATS


def metrics(age_idx, female, race_idx):
    """age_idx: 0..4 정수 배열, female: 0/1, race_idx: 0..4 또는 -1(미상)."""
    n = len(age_idx)
    ap = np.bincount(age_idx, minlength=5) / n * 100
    out = {
        "TV_age": 0.5 * np.abs(ap - np.array([C.HINTS_AGE[k] for k in AGE_LABELS])).sum(),
        "female": 100 * female.mean(),
        "female_diff": 100 * female.mean() - C.HINTS_FEMALE,
        "age65plus": ap[3] + ap[4],
    }
    known = race_idx >= 0
    if known.any():
        rp = np.bincount(race_idx[known], minlength=5) / known.sum() * 100
        out["TV_race"] = 0.5 * np.abs(rp - np.array([C.HINTS_RACE[k] for k in RACE_CATS])).sum()
    return out


def prep(df):
    age = pd.cut(df["age"], C.AGE_BINS, labels=False, right=False)
    ok = age.notna()
    d = df[ok]
    age_idx = age[ok].astype(int).values
    female = (d["female"] == 1).astype(int).values
    if "race5" in d.columns:
        m = {k: i for i, k in enumerate(RACE_CATS)}
        race_idx = d["race5"].map(m).fillna(-1).astype(int).values
    else:
        race_idx = np.full(len(d), -1)
    return age_idx, female, race_idx


def boot(df, B, rng):
    age_idx, female, race_idx = prep(df)
    n = len(age_idx)
    point = metrics(age_idx, female, race_idx)
    draws = {k: [] for k in point}
    for _ in range(B):
        i = rng.integers(0, n, n)
        m = metrics(age_idx[i], female[i], race_idx[i])
        for k in point:
            if k in m:
                draws[k].append(m[k])
    rows = []
    for k, v in point.items():
        lo, hi = np.percentile(draws[k], [2.5, 97.5])
        rows.append({"metric": k, "estimate": v, "ci_lo": lo, "ci_hi": hi, "n": n, "B": B})
    return rows


def coverage(df, db, group, min_cell, joint):
    age = pd.cut(df["age"], C.AGE_BINS, labels=AGE_LABELS, right=False)
    sex = np.where(df["female"] == 1, "F", "M")
    t = pd.crosstab(age, sex).reindex(index=AGE_LABELS, columns=["F", "M"]).fillna(0).astype(int)
    n = int(t.values.sum())
    rows = []
    for ag in AGE_LABELS:
        for sx in ("F", "M"):
            c = int(t.loc[ag, sx])
            r = {"db": db, "group": group, "age": ag, "sex": sx, "n_cell": c,
                 "db_pct": 100 * c / n, "below_min": c < min_cell}
            if joint is not None:
                ref = joint[(joint["age"] == ag) & (joint["sex"] == sx)]["pct"].iloc[0]
                r["hints_pct"] = ref
                r["ratio_db_over_hints"] = r["db_pct"] / ref if ref > 0 else np.nan
                r["hints_pct_covered"] = ref if c >= min_cell else 0.0
            rows.append(r)
    return rows


def apply_reference(csv_path, year, group):
    """HINTS 기준값을 hints_reference_long.csv의 (year, group)으로 바꾼다(민감도 분석용)."""
    d = pd.read_csv(csv_path)
    d = d[(d["year"] == year) & (d["group"] == group)]
    if d.empty:
        raise SystemExit(f"reference not found: year={year}, group={group}")
    g = lambda v: d[d["variable"] == v].set_index("category")["pct"]
    age, race, fem = g("age"), g("race"), g("female")
    C.HINTS_AGE = {k: float(age[k]) for k in AGE_LABELS}
    C.HINTS_RACE = {k: float(race[k]) for k in RACE_CATS}
    C.HINTS_FEMALE = float(fem.iloc[0])
    return f"reference: HINTS {year} {group}; age={C.HINTS_AGE}; race={C.HINTS_RACE}; female={C.HINTS_FEMALE:.1f}"


def mimic_first_ecg(mimiciv, record_list, notes):
    """MIMIC-IV-ECG: 환자당 첫 ECG(성인), 환경·인종 부여. BMI는 CI 대상에서 제외(이 스크립트는 나이·성별·인종만).
    규칙은 mimic_table1_in_hospital.py와 동일."""
    import mimic_table1_in_hospital as M
    hosp = os.path.join(mimiciv, "hosp")
    pt = pd.read_csv(os.path.join(hosp, "patients.csv.gz"), usecols=["subject_id", "gender", "anchor_age", "anchor_year"])
    adm = pd.read_csv(os.path.join(hosp, "admissions.csv.gz"))
    ecg = pd.read_csv(record_list, usecols=["subject_id", "study_id", "ecg_time"])
    ecg["ecg_time"] = pd.to_datetime(ecg["ecg_time"], errors="coerce")
    ecg = ecg.dropna(subset=["ecg_time"])
    df = ecg.merge(pt, on="subject_id", how="inner")
    df["age"] = df["anchor_age"] + (df["ecg_time"].dt.year - df["anchor_year"])
    df = df[df["age"] >= 18].copy()
    df = df.merge(M.load_race(adm), on="subject_id", how="left")
    df = M.attach_setting(df, adm)
    df = df.sort_values("ecg_time").drop_duplicates("subject_id").reset_index(drop=True)
    df["female"] = (df["gender"] == "F").astype(int)
    df["setting"] = df["setting"].str.lower()
    notes.append(f"MIMIC-IV-ECG: first adult ECG per patient, {len(df)} patients")
    return df


def main():
    ap = argparse.ArgumentParser()
    h = "/home/this14"
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--min_cell", type=int, default=100)
    ap.add_argument("--hints_joint", default=None)
    ap.add_argument("--echonext", default=f"{h}/us_representativeness/data/EchoNext/echonext_metadata_100k_v1.1.1.csv")
    ap.add_argument("--georgia", default=f"{h}/us_representativeness/data/georgia")
    ap.add_argument("--snuh", default=f"{h}/LYDUS_SNUH_ECG_ANONYMIZED_50K_0512.csv")
    ap.add_argument("--ptbxl", default=f"{h}/data/ptb-xl/ptbxl_database.csv")
    ap.add_argument("--mimiciv", default=None, help="MIMIC-IV 루트(hosp 상위). 주면 MIMIC도 계산")
    ap.add_argument("--record_list", default=None, help="MIMIC-IV-ECG record_list.csv")
    ap.add_argument("--ref_csv", default=None, help="hints_reference_long.csv. 주면 기준값을 --ref_year/--ref_group으로 교체(민감도 분석)")
    ap.add_argument("--ref_year", type=int, default=2024)
    ap.add_argument("--ref_group", default="wearable_user", choices=["wearable_user", "non_user", "all_adults"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    notes = []
    if a.ref_csv:
        msg = apply_reference(a.ref_csv, a.ref_year, a.ref_group)
        notes.append(msg)
        print(msg)
    joint = None
    if a.hints_joint and os.path.exists(a.hints_joint):
        j = pd.read_csv(a.hints_joint)
        if a.ref_year == 2024:  # 연령×성별 결합분포는 2024만 있다
            joint = j[j["group"] == a.ref_group]
        else:
            print("coverage skipped: age-by-sex joint is available for 2024 only")
    sets = [("EchoNext", "Overall", C.echonext(a.echonext, notes)),
            ("Georgia", "Overall", C.georgia(a.georgia, notes)),
            ("SNUH", "Overall", C.snuh(a.snuh, notes)),
            ("PTB-XL", "Overall", C.ptbxl(a.ptbxl, notes))]
    e = sets[0][2]
    for s in ("emergency", "inpatient", "outpatient", "procedural"):
        sub = e[e["setting"] == s]
        if len(sub):
            sets.append(("EchoNext", s, sub))
    if a.mimiciv and a.record_list:
        mm = mimic_first_ecg(a.mimiciv, a.record_list, notes)
        sets.append(("MIMIC", "Overall", mm))
        for s_, lab in (("ed", "ED"), ("inpatient", "Inpatient"), ("outside admission", "Outside admission")):
            sub = mm[mm["setting"] == s_]
            if len(sub):
                sets.append(("MIMIC", lab, sub))
    ci_rows, cov_rows = [], []
    for db, grp, df in sets:
        for r in boot(df, a.B, rng):
            ci_rows.append({"db": db, "group": grp, **r})
        cov_rows += coverage(df, db, grp, a.min_cell, joint)
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(os.path.join(a.out, "bootstrap_ci_long.csv"), index=False)
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(os.path.join(a.out, "age_sex_coverage_long.csv"), index=False)
    if joint is not None:
        summ = (cov.groupby(["db", "group"]).agg(hints_pct_covered=("hints_pct_covered", "sum"),
                                                 cells_below_min=("below_min", "sum")).round(1))
        summ.to_csv(os.path.join(a.out, "age_sex_coverage_summary.csv"))
        print(summ.to_string())
    ci["text"] = ci.apply(lambda r: f"{r['estimate']:.1f} ({r['ci_lo']:.1f}-{r['ci_hi']:.1f})", axis=1)
    wide = ci.pivot_table(index=["db", "group", "n"], columns="metric", values="text", aggfunc="first")
    wide.to_csv(os.path.join(a.out, "bootstrap_ci_summary.csv"))
    with open(os.path.join(a.out, "bootstrap_notes.txt"), "w") as f:
        f.write("\n".join(notes) + f"\nB={a.B}, seed={a.seed}, min_cell={a.min_cell}\n")
        f.write("CI = percentile bootstrap over patients (rows). Georgia/SNUH rows are ECGs, not verified patients.\n")
        f.write("HINTS reference values treated as fixed.\n")
    print(wide.to_string())


if __name__ == "__main__":
    main()
