#!/usr/bin/env python3
"""No.64: HINTS 5 Cycle 4(2020), HINTS 6(2022), HINTS 7(2024)에서 웨어러블 사용자·비사용자의
나이대·성별·인종·BMI 분포(가중, 95% CI)를 계산한다.

사용법:
    python3 hints_reference.py --h5 <hints5_cycle4_public.sav> --h6 <hints6_public.rda> --h7 <hints7_public.rda> --out <폴더>

규칙(결과를 보기 전에 고정):
  - 웨어러블 사용자 = 최근 12개월 사용(Yes). 비사용자 = 그 외(2024는 '최근 12개월 아님'+'사용한 적 없음' 포함)
  - 가중치 PERSON_FINWT0, 복제가중치 PERSON_FINWT1~50. 분산 = (R-1)/R * sum (theta_r - theta)^2, R=50 (jackknife)
  - 분모는 변수별로 결측(Not Ascertained, Web partial 등)을 제외
  - 인종 5범주(RaceEthn5): White / Black / Hispanic / Asian / Other (모두 non-Hispanic, Hispanic 별도)
  - BMI는 HINTS 제공 변수(자가보고 키·체중 기반). <18.5, 18.5-24.9, 25-29.9, >=30
  - 문항 문구가 회차마다 달라 연도 간 추세는 코드북 확인 후에만 해석(2024는 '사용한 적 없음'이 따로 있음)
"""
import argparse
import os
import numpy as np
import pandas as pd
import pyreadr
import pyreadstat

R = 50
AGE = ["18-34", "35-49", "50-64", "65-74", "75+"]
RACE_MAP = {"Non-Hispanic White": "White", "Non-Hispanic Black or African American": "Black",
            "Hispanic": "Hispanic", "Non-Hispanic Asian": "Asian", "Non-Hispanic Other": "Other"}
RACE_ORDER = ["White", "Black", "Hispanic", "Asian", "Other"]
BMI_BINS = [0, 18.5, 25, 30, 1000]
BMI_LABELS = ["<18.5", "18.5-24.9", "25-29.9", ">=30"]


def load(path, kind):
    if kind == "sav":
        d, m = pyreadstat.read_sav(path)
        lab = m.variable_value_labels
        for v in ["WearableDevTrackHealth", "AgeGrpB", "BirthGender", "RaceEthn5"]:
            d[v] = d[v].map(lab[v]).astype("string")
        d["sex"] = d["BirthGender"]
        d["wear"] = d["WearableDevTrackHealth"]
    else:
        d = pyreadr.read_r(path)["public"]
        wcol = "WearableDevTrackHealth2" if "WearableDevTrackHealth2" in d.columns else "WearableDevTrackHealth"
        d["wear"] = d[wcol].astype("string")
        d["sex"] = d["BirthSex"].astype("string") if "BirthSex" in d.columns else d["BirthGender"].astype("string")
        d["AgeGrpB"] = d["AgeGrpB"].astype("string")
        d["RaceEthn5"] = d["RaceEthn5"].astype("string")
    d["age_cat"] = d["AgeGrpB"].where(d["AgeGrpB"].isin(AGE))
    d["female"] = np.where(d["sex"] == "Female", 1.0, np.where(d["sex"] == "Male", 0.0, np.nan))
    d["race5"] = d["RaceEthn5"].map(RACE_MAP)
    bmi = pd.to_numeric(d["BMI"], errors="coerce")
    bmi = bmi.where((bmi >= 10) & (bmi <= 80))
    d["bmi"] = bmi
    d["bmi_cat"] = pd.cut(d["bmi"], BMI_BINS, labels=BMI_LABELS, right=False).astype("string")
    d["user"] = np.where(d["wear"] == "Yes", 1, np.where(d["wear"].isin(["No", "No, not in the past 12 months",
                         "I have never used an electronic wearable device"]), 0, np.nan))
    w = ["PERSON_FINWT0"] + [f"PERSON_FINWT{i}" for i in range(1, R + 1)]
    for c in w:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d, w


def wprop(d, w, var, cats, mask):
    sub = d[mask & d[var].notna()]
    out = {}
    for k in cats:
        ind = (sub[var] == k).astype(float).values
        est = np.sum(sub[w[0]].values * ind) / np.sum(sub[w[0]].values)
        reps = np.array([np.sum(sub[c].values * ind) / np.sum(sub[c].values) for c in w[1:]])
        se = np.sqrt((R - 1) / R * np.sum((reps - est) ** 2))
        out[k] = (100 * est, 100 * (est - 1.96 * se), 100 * (est + 1.96 * se))
    return out, len(sub)


def wmean(d, w, var, mask):
    sub = d[mask & d[var].notna()]
    x = sub[var].values
    est = np.sum(sub[w[0]].values * x) / np.sum(sub[w[0]].values)
    reps = np.array([np.sum(sub[c].values * x) / np.sum(sub[c].values) for c in w[1:]])
    se = np.sqrt((R - 1) / R * np.sum((reps - est) ** 2))
    return 100 * est, 100 * (est - 1.96 * se), 100 * (est + 1.96 * se), len(sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--h6", required=True)
    ap.add_argument("--h7", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for year, path, kind in ((2020, a.h5, "sav"), (2022, a.h6, "rda"), (2024, a.h7, "rda")):
        d, w = load(path, kind)
        for label, mask in (("wearable_user", d["user"] == 1), ("non_user", d["user"] == 0),
                            ("all_adults", d["user"].notna())):
            n_group = int(mask.sum())
            for var, cats, name in (("age_cat", AGE, "age"), ("race5", RACE_ORDER, "race"),
                                    ("bmi_cat", BMI_LABELS, "bmi")):
                res, n = wprop(d, w, var, cats, mask)
                for k, (e, lo, hi) in res.items():
                    rows.append({"year": year, "group": label, "variable": name, "category": k,
                                 "pct": e, "ci_lo": lo, "ci_hi": hi, "n_valid": n, "n_group": n_group})
            e, lo, hi, n = wmean(d, w, "female", mask)
            rows.append({"year": year, "group": label, "variable": "female", "category": "",
                         "pct": e, "ci_lo": lo, "ci_hi": hi, "n_valid": n, "n_group": n_group})
        # 전체 성인 중 웨어러블 사용률
        sub = d[d["user"].notna()]
        ind = sub["user"].values
        est = np.sum(sub[w[0]].values * ind) / np.sum(sub[w[0]].values)
        reps = np.array([np.sum(sub[c].values * ind) / np.sum(sub[c].values) for c in w[1:]])
        se = np.sqrt((R - 1) / R * np.sum((reps - est) ** 2))
        rows.append({"year": year, "group": "all_adults", "variable": "wearable_use_rate", "category": "",
                     "pct": 100 * est, "ci_lo": 100 * (est - 1.96 * se), "ci_hi": 100 * (est + 1.96 * se),
                     "n_valid": len(sub), "n_group": len(sub)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.out, "hints_reference_long.csv"), index=False)
    print(out[(out["group"] == "wearable_user") & (out["year"] == 2024)].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
