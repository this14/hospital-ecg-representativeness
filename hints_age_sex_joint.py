#!/usr/bin/env python3
"""No.64 RQ2 기준값: HINTS 웨어러블 사용자의 나이대 x 성별 결합분포(가중, jackknife 95% CI).

hints_reference.py의 load()를 그대로 재사용한다(같은 폴더에 두고 실행).
사용법:
    python3 hints_age_sex_joint.py --h7 <hints7_public.rda> --out <폴더> [--year 2024]
출력: hints_age_sex_joint.csv (year, group, age, sex, pct, ci_lo, ci_hi, n_valid)
  pct = 해당 집단(웨어러블 사용자 또는 전체 성인) 안에서 그 셀이 차지하는 가중 비율(합 100)
"""
import argparse
import os
import numpy as np
import pandas as pd
from hints_reference import load, AGE, R

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h7", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d, w = load(a.h7, "rda")
    d["sexcat"] = np.where(d["female"] == 1, "F", np.where(d["female"] == 0, "M", None))
    rows = []
    for label, mask in (("wearable_user", d["user"] == 1), ("all_adults", d["user"].notna())):
        sub = d[mask & d["age_cat"].notna() & d["sexcat"].notna()]
        for ag in AGE:
            for sx in ("F", "M"):
                ind = ((sub["age_cat"] == ag) & (sub["sexcat"] == sx)).astype(float).values
                est = np.sum(sub[w[0]].values * ind) / np.sum(sub[w[0]].values)
                reps = np.array([np.sum(sub[c].values * ind) / np.sum(sub[c].values) for c in w[1:]])
                se = np.sqrt((R - 1) / R * np.sum((reps - est) ** 2))
                rows.append({"year": 2024, "group": label, "age": ag, "sex": sx, "pct": 100 * est,
                             "ci_lo": 100 * (est - 1.96 * se), "ci_hi": 100 * (est + 1.96 * se), "n_valid": len(sub)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.out, "hints_age_sex_joint.csv"), index=False)
    print(out.round(1).to_string(index=False))

if __name__ == "__main__":
    main()
