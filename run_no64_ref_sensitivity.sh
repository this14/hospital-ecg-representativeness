#!/bin/bash
# 기준 인구 민감도 분석: HINTS 2020/2022/2024 웨어러블 사용자, 2024 전체 성인.  사용: bash run_no64_ref_sensitivity.sh
cd /home/this14
REF=$(find /home/this14 -name hints_reference_long.csv 2>/dev/null | head -1)
if [ -z "$REF" ]; then
  echo "hints_reference_long.csv not found under /home/this14. Put it there (or run hints_reference.py again) and rerun."
  exit 1
fi
echo "using reference file: $REF"
for spec in "2020 wearable_user" "2022 wearable_user" "2024 wearable_user" "2024 all_adults"; do
  set -- $spec
  out=/home/this14/no64_out_ref_$1_$2
  python3 no64_bootstrap_ci.py --out $out --B 500 \
    --ref_csv $REF --ref_year $1 --ref_group $2 \
    --hints_joint /home/this14/no64_out/hints_age_sex_joint.csv \
    --mimiciv /home/this14/mimiciv --record_list /home/this14/mimic-iv-ecg/record_list.csv 2> $out.err > /dev/null || { echo "FAILED: $1 $2 (see $out.err)"; tail -5 $out.err; continue; }
  echo "=== reference: HINTS $1 $2 ==="
  python3 -c "import pandas as pd; d=pd.read_csv('$out/bootstrap_ci_summary.csv'); print(d[['db','group','n','TV_age','age65plus','female_diff']].to_string(index=False))"
done
