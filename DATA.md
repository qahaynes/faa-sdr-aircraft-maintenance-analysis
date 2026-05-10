# Data Documentation

## Source

The repository uses the Federal Aviation Administration (FAA) Service Difficulty Report (SDR) public data by year.

Default CSV endpoint used by the script:

```text
https://external.apic4e.faa.gov/sdrs/retrieve/SDR-2024.csv
```

Official FAA page:

```text
https://www.faa.gov/av-info/download_SDR
```

## Why the raw data are not committed

The raw FAA SDR CSV files are public and may be large. To keep the repository lightweight and reproducible, the analysis script downloads the selected year directly from the FAA and stores a local cache in `data/raw/`.

The `.gitignore` file excludes downloaded CSV files so they are not accidentally committed.

## Variables used

The analysis uses the following FAA SDR fields:

- `AircraftTotalTime`: total aircraft time in hours.
- `PartCondition`: condition of the reported part.

The script filters `PartCondition` to:

- `CRACKED`
- `CORRODED`

A regression indicator is then created:

- `Cracked = 1` for `CRACKED`
- `Cracked = 0` for `CORRODED`

## Cleaning rules

The script:

1. Converts `PartCondition` to uppercase text and removes leading/trailing spaces.
2. Converts `AircraftTotalTime` to numeric.
3. Keeps only records coded `CRACKED` or `CORRODED`.
4. Drops records with missing or nonnumeric `AircraftTotalTime`.
5. Removes negative `AircraftTotalTime` values.
6. Requires at least two observations in each comparison group.

## Reproducibility note

FAA data files can be updated or corrected after download. To reproduce a prior run exactly, preserve the downloaded file used in the analysis or document the date when the data were retrieved.
