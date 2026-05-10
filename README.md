# FAA SDR Aircraft Maintenance Analysis

This repository reproduces the analysis for:

**Aircraft Maintenance Conditions and Total Aircraft Time: An Analysis of FAA Service Difficulty Report Data**

The project evaluates whether FAA Service Difficulty Report (SDR) records coded `CRACKED` have a different mean `AircraftTotalTime` than records coded `CORRODED`. The analysis uses the actual FAA SDR calendar-year CSV file downloaded from the FAA SDR public data source.

## Data source

The analysis script downloads the FAA SDR CSV file directly from the FAA year-based SDR download endpoint:

```text
https://external.apic4e.faa.gov/sdrs/retrieve/SDR-2024.csv
```

The default year is 2024 because that is the year used for the course analysis. To use another available FAA SDR year, pass a different year value, such as `--year 2025`.

The raw CSV file is **not committed to the repository** because the FAA file is public, can be large, and should be downloaded directly from the official source for reproducibility. When the script runs, it caches the raw file under:

```text
data/raw/SDR-YYYY.csv
```

## Research question

Is there a statistically significant difference in mean `AircraftTotalTime` between FAA SDR records with `PartCondition` coded `CRACKED` and those coded `CORRODED`?

## Variables

| Variable | Role | Description |
|---|---|---|
| `AircraftTotalTime` | Dependent variable | Total aircraft operating time in hours |
| `PartCondition` | Grouping variable | Filtered to `CRACKED` and `CORRODED` |
| `Cracked` | Regression indicator | `1 = CRACKED`; `0 = CORRODED` |

## Methods

The script performs:

1. Data download and local caching from the FAA SDR CSV endpoint.
2. Cleaning of `PartCondition` and `AircraftTotalTime`.
3. Complete-case filtering for `CRACKED` and `CORRODED` records.
4. Descriptive statistics by condition.
5. Welch two-sample t test.
6. Point-biserial correlation.
7. Dummy-coded OLS regression.
8. Chart generation for report counts and mean aircraft total time.

Welch's t test is used because it does not assume equal variances across the two condition groups.

## Repository structure

```text
faa-sdr-aircraft-maintenance-analysis/
├── README.md
├── DATA.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── src/
│   └── analyze_sdr.py
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
└── outputs/
    ├── figures/
    │   └── .gitkeep
    └── tables/
        └── .gitkeep
```

## Setup

Create and activate a virtual environment.

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the analysis with actual FAA data

```bash
python src/analyze_sdr.py --year 2024
```

The script will download and cache the actual FAA SDR CSV for 2024, then produce output files in the `outputs/` folder.

## Use a locally downloaded FAA file

If you manually download the CSV from the FAA first, run:

```bash
python src/analyze_sdr.py --input data/raw/SDR-2024.csv
```

## Refresh the cached FAA data

```bash
python src/analyze_sdr.py --year 2024 --refresh
```

## Expected outputs

After the script runs, it writes:

```text
outputs/tables/descriptive_statistics.csv
outputs/tables/summary_results.csv
outputs/tables/regression_coefficients.csv
outputs/figures/report_counts_by_part_condition.png
outputs/figures/mean_aircraft_total_time_by_part_condition.png
```

It also prints the main statistical results to the terminal.
