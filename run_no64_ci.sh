#!/bin/bash
# 사용: bash run_no64_ci.sh   (홈 디렉터리에서)
cd /home/this14
python3 no64_bootstrap_ci.py \
  --out /home/this14/no64_out \
  --hints_joint /home/this14/no64_out/hints_age_sex_joint.csv \
  --mimiciv /home/this14/mimiciv \
  --record_list /home/this14/mimic-iv-ecg/record_list.csv
