#!/bin/bash
# min_cell 민감도 분석 (50, 200). 사용: bash run_no64_sensitivity.sh
cd /home/this14
for m in 50 200; do
  python3 no64_bootstrap_ci.py \
    --out /home/this14/no64_out_min$m \
    --min_cell $m --B 200 \
    --hints_joint /home/this14/no64_out/hints_age_sex_joint.csv \
    --mimiciv /home/this14/mimiciv \
    --record_list /home/this14/mimic-iv-ecg/record_list.csv > /dev/null
  echo "=== min_cell=$m ==="
  cat /home/this14/no64_out_min$m/age_sex_coverage_summary.csv
done
