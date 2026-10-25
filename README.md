# hospital-ecg-representativeness

Code and aggregate results for the manuscript "How far are hospital ECG databases from the people who will wear the device? A quantitative representativeness gap analysis against US wearable users" (C. Lee).

The analysis compares the age, sex and race/ethnicity distribution of five hospital ECG resources with US adults who use wearable devices (HINTS 7, 2024), using total variation distance (TV), patient-level percentile bootstrap intervals, age-by-sex coverage and cell-wise relative representation.

**No patient-level data are included.** Only aggregate tables derived from public HINTS data and summary results are in this repository.

## Data (obtain separately, follow each licence)

| Resource | Where | Access |
|---|---|---|
| HINTS 5 Cycle 4 (2020), HINTS 6 (2022), HINTS 7 (2024) | National Cancer Institute, hints.cancer.gov (R/SPSS public-use files) | public |
| MIMIC-IV-ECG v1.0 and MIMIC-IV | PhysioNet (doi:10.13026/4nqg-sb35 for MIMIC-IV-ECG) | credentialed |
| EchoNext-Mini | PhysioNet | credentialed |
| PhysioNet/CinC Challenge 2020 (Georgia) | PhysioNet | open |
| PTB-XL | PhysioNet | open |
| SNUH LYDUS ECG 50K v1.1.1 | Korea Health Data Platform, doi 10.23421/2025.0704 (CC-BY-4.0) | registered users |

## Files

- `hints_reference.py`: weighted HINTS distributions for wearable users, non-users and all adults (jackknife replicate weights); writes `hints_reference_long.csv` (included).
- `hints_age_sex_joint.py`: HINTS 2024 age-by-sex joint distribution (10 cells) with confidence intervals.
- `mimic_table1_in_hospital.py`: MIMIC-IV-ECG demographics and care-setting derivation.
- `no64_compare_all_dbs.py`: loaders for EchoNext-Mini, Georgia, SNUH and PTB-XL, and the point estimates.
- `no64_bootstrap_ci.py`: bootstrap intervals, coverage and cell-wise relative representation; options for the minimum cell size and for alternative reference populations.
- `run_no64_ci.sh`, `run_no64_sensitivity.sh`, `run_no64_ref_sensitivity.sh`: the runs reported in the paper (main analysis; minimum cell size 50 and 200; reference populations).
- `results/bootstrap_ci_summary.csv`: main bootstrap results.

## Reproduce

Python 3.12 with pandas, numpy, pyreadr and pyreadstat. Paths in the scripts default to the author's server (`/home/this14/...`); override them with the command-line options (`--echonext`, `--georgia`, `--snuh`, `--ptbxl`, `--mimiciv`, `--record_list`, `--out`) or edit the `run_*.sh` files.

```bash
python3 hints_reference.py --h5 <hints5_cycle4_public.sav> --h6 <hints6_public.rda> --h7 <hints7_public.rda> --out <dir>
python3 hints_age_sex_joint.py --h7 <hints7_public.rda> --out <dir>
python3 no64_bootstrap_ci.py --out <dir> --hints_joint <dir>/hints_age_sex_joint.csv \
    --mimiciv <mimiciv_root> --record_list <mimic-iv-ecg>/record_list.csv
```

Bootstrap settings: B = 2000, seed 20261024, HINTS reference values treated as fixed.

## Notes

- The SNUH resource is a selected abnormal-findings subset of a larger collection; see the manuscript.
- The cell-wise relative representation, minimum cell size sensitivity and alternative reference populations were added after the main results were seen (post hoc).

## Licence

Code: MIT (see `LICENSE`). Data remain under the licences of their sources.
