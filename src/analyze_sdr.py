#!/usr/bin/env python3
"""
Analyze FAA Service Difficulty Report data by part condition.

Default analysis:
- Data source: FAA SDR calendar-year CSV.
- Dependent variable: AircraftTotalTime.
- Grouping variable: PartCondition, filtered to CRACKED and CORRODED.
- Statistical tests: Welch two-sample t test, point-biserial correlation,
  and dummy-coded OLS regression.

Run:
    python src/analyze_sdr.py --year 2024

Optional local file:
    python src/analyze_sdr.py --input data/raw/SDR-2024.csv
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


FAA_SDR_URL_TEMPLATE = "https://external.apic4e.faa.gov/sdrs/retrieve/SDR-{year}.csv"
DEFAULT_YEAR = 2024


def build_faa_sdr_url(year: int) -> str:
    """Build the FAA SDR CSV URL for a calendar year."""
    return FAA_SDR_URL_TEMPLATE.format(year=year)


def download_faa_sdr_csv(year: int, destination: Path, refresh: bool = False) -> Path:
    """
    Download and cache the FAA SDR CSV file for the selected year.

    Parameters
    ----------
    year:
        FAA SDR calendar year.
    destination:
        Local path where the CSV should be stored.
    refresh:
        If True, download again even if the file already exists.

    Returns
    -------
    Path
        Path to the local cached CSV file.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not refresh:
        print(f"Using cached FAA SDR file: {destination}")
        return destination

    url = build_faa_sdr_url(year)
    print(f"Downloading FAA SDR data from: {url}")
    print(f"Saving to: {destination}")

    try:
        urllib.request.urlretrieve(url, destination)
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not download the FAA SDR CSV file. "
            "Check your internet connection, confirm the selected year exists on "
            "the FAA SDR download page, or download the CSV manually and use --input."
        ) from exc

    return destination


def load_sdr_data(
    input_path: Optional[Path],
    year: int,
    data_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    """Load FAA SDR data from a local input path or from the FAA download endpoint."""
    if input_path is not None:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        source = input_path
    else:
        source = download_faa_sdr_csv(
            year=year,
            destination=data_dir / "raw" / f"SDR-{year}.csv",
            refresh=refresh,
        )

    print(f"Reading data from: {source}")
    return pd.read_csv(source, low_memory=False)


def prepare_analysis_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and filter the SDR data for the CRACKED vs CORRODED analysis.

    Returns a DataFrame with:
    - PartCondition
    - AircraftTotalTime
    - Cracked
    """
    required_columns = {"PartCondition", "AircraftTotalTime"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required column(s): {sorted(missing_columns)}")

    analysis = pd.DataFrame(
        {
            "PartCondition": df["PartCondition"].astype("string").str.upper().str.strip(),
            "AircraftTotalTime": pd.to_numeric(df["AircraftTotalTime"], errors="coerce"),
        }
    )

    analysis = analysis.loc[
        analysis["PartCondition"].isin(["CRACKED", "CORRODED"]),
        ["PartCondition", "AircraftTotalTime"],
    ].copy()

    analysis = analysis.dropna(subset=["AircraftTotalTime"])
    analysis = analysis.loc[analysis["AircraftTotalTime"] >= 0].copy()

    if analysis.empty:
        raise ValueError(
            "No usable records remained after filtering to CRACKED/CORRODED "
            "and valid AircraftTotalTime."
        )

    group_counts = analysis["PartCondition"].value_counts()

    missing_groups = {"CRACKED", "CORRODED"}.difference(group_counts.index)
    if missing_groups:
        raise ValueError(f"Missing comparison group(s): {sorted(missing_groups)}")

    if (group_counts[["CRACKED", "CORRODED"]] < 2).any():
        raise ValueError("Each group must contain at least two usable records.")

    analysis["Cracked"] = np.where(analysis["PartCondition"] == "CRACKED", 1, 0)

    return analysis


def descriptive_statistics(analysis: pd.DataFrame) -> pd.DataFrame:
    """Create descriptive statistics by PartCondition."""
    desc = (
        analysis.groupby("PartCondition", observed=True)["AircraftTotalTime"]
        .agg(
            n="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max",
        )
        .sort_index()
    )

    percentages = (
        analysis["PartCondition"]
        .value_counts(normalize=True)
        .mul(100)
        .rename("percent_of_subset")
    )

    desc = desc.join(percentages)
    desc = desc[["n", "percent_of_subset", "mean", "median", "std", "min", "max"]]

    return desc


def welch_t_test(analysis: pd.DataFrame) -> dict[str, float]:
    """
    Conduct a Welch two-sample t test for CRACKED minus CORRODED.

    The sign convention is:
    mean difference = mean(CRACKED) - mean(CORRODED)
    """
    cracked = analysis.loc[
        analysis["PartCondition"] == "CRACKED",
        "AircraftTotalTime",
    ]

    corroded = analysis.loc[
        analysis["PartCondition"] == "CORRODED",
        "AircraftTotalTime",
    ]

    n_cracked = cracked.count()
    n_corroded = corroded.count()

    mean_cracked = cracked.mean()
    mean_corroded = corroded.mean()
    mean_difference = mean_cracked - mean_corroded

    var_cracked = cracked.var(ddof=1)
    var_corroded = corroded.var(ddof=1)

    se = np.sqrt((var_cracked / n_cracked) + (var_corroded / n_corroded))

    df_numerator = ((var_cracked / n_cracked) + (var_corroded / n_corroded)) ** 2
    df_denominator = (
        ((var_cracked / n_cracked) ** 2 / (n_cracked - 1))
        + ((var_corroded / n_corroded) ** 2 / (n_corroded - 1))
    )
    welch_df = df_numerator / df_denominator

    t_statistic = mean_difference / se
    p_value = 2 * stats.t.sf(abs(t_statistic), df=welch_df)

    t_critical = stats.t.ppf(0.975, df=welch_df)
    ci_lower = mean_difference - t_critical * se
    ci_upper = mean_difference + t_critical * se

    effect_r = np.sign(t_statistic) * np.sqrt(
        (t_statistic**2) / ((t_statistic**2) + welch_df)
    )

    return {
        "cracked_n": float(n_cracked),
        "corroded_n": float(n_corroded),
        "cracked_mean": float(mean_cracked),
        "corroded_mean": float(mean_corroded),
        "mean_difference_cracked_minus_corroded": float(mean_difference),
        "welch_standard_error": float(se),
        "welch_degrees_of_freedom": float(welch_df),
        "t_statistic": float(t_statistic),
        "p_value": float(p_value),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "effect_r_from_t": float(effect_r),
        "effect_r_squared_from_t": float(effect_r**2),
    }


def regression_analysis(analysis: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """
    Fit the dummy-coded OLS regression.

    Model:
        AircraftTotalTime = beta0 + beta1(Cracked)

    Because Cracked is binary:
    - beta0 is the CORRODED group mean.
    - beta1 is the CRACKED minus CORRODED mean difference.
    """
    model = smf.ols("AircraftTotalTime ~ Cracked", data=analysis).fit()

    coef_table = pd.DataFrame(
        {
            "term": model.params.index,
            "estimate": model.params.values,
            "ols_standard_error": model.bse.values,
            "t_value": model.tvalues.values,
            "p_value": model.pvalues.values,
            "ci_95_lower": model.conf_int()[0].values,
            "ci_95_upper": model.conf_int()[1].values,
        }
    )

    point_biserial_r = float(analysis["AircraftTotalTime"].corr(analysis["Cracked"]))
    r_squared = float(model.rsquared)

    return coef_table, point_biserial_r, r_squared


def write_outputs(
    analysis: pd.DataFrame,
    desc: pd.DataFrame,
    welch_results: dict[str, float],
    coef_table: pd.DataFrame,
    point_biserial_r: float,
    regression_r_squared: float,
    output_dir: Path,
    create_plots: bool = True,
) -> None:
    """Write tables and figures to the output directory."""
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    desc.round(4).to_csv(tables_dir / "descriptive_statistics.csv")

    summary_rows = [{"statistic": key, "value": value} for key, value in welch_results.items()]
    summary_rows.extend(
        [
            {"statistic": "point_biserial_r", "value": point_biserial_r},
            {"statistic": "regression_r_squared", "value": regression_r_squared},
        ]
    )

    pd.DataFrame(summary_rows).to_csv(tables_dir / "summary_results.csv", index=False)
    coef_table.to_csv(tables_dir / "regression_coefficients.csv", index=False)

    if not create_plots:
        return

    condition_order = ["CORRODED", "CRACKED"]

    counts = analysis["PartCondition"].value_counts().reindex(condition_order)

    plt.figure()
    counts.plot(kind="bar")
    plt.xlabel("Part condition")
    plt.ylabel("Number of reports")
    plt.title("Report Counts by PartCondition")
    plt.tight_layout()
    plt.savefig(figures_dir / "report_counts_by_part_condition.png", dpi=300)
    plt.close()

    means = (
        analysis.groupby("PartCondition", observed=True)["AircraftTotalTime"]
        .mean()
        .reindex(condition_order)
    )

    plt.figure()
    means.plot(kind="bar")
    plt.xlabel("Part condition")
    plt.ylabel("Mean aircraft total time, hours")
    plt.title("Mean AircraftTotalTime by PartCondition")
    plt.tight_layout()
    plt.savefig(figures_dir / "mean_aircraft_total_time_by_part_condition.png", dpi=300)
    plt.close()


def print_report(
    desc: pd.DataFrame,
    welch_results: dict[str, float],
    coef_table: pd.DataFrame,
    point_biserial_r: float,
    regression_r_squared: float,
    output_dir: Path,
) -> None:
    """Print a concise terminal report."""
    print("\nDescriptive statistics")
    print(desc.round(2).to_string())

    print("\nWelch two-sample t test")
    print(
        f"CRACKED: n = {welch_results['cracked_n']:,.0f}, "
        f"mean = {welch_results['cracked_mean']:,.2f} hours"
    )
    print(
        f"CORRODED: n = {welch_results['corroded_n']:,.0f}, "
        f"mean = {welch_results['corroded_mean']:,.2f} hours"
    )
    print(
        "Mean difference, CRACKED - CORRODED = "
        f"{welch_results['mean_difference_cracked_minus_corroded']:,.2f} hours"
    )
    print(f"SE = {welch_results['welch_standard_error']:,.2f}")
    print(
        f"t({welch_results['welch_degrees_of_freedom']:,.2f}) = "
        f"{welch_results['t_statistic']:,.2f}"
    )
    print(f"p = {welch_results['p_value']:.6f}")
    print(
        "95% CI = "
        f"[{welch_results['ci_95_lower']:,.2f}, "
        f"{welch_results['ci_95_upper']:,.2f}]"
    )

    print("\nDummy-coded regression")
    print(coef_table.round(4).to_string(index=False))
    print(f"\nPoint-biserial correlation: r = {point_biserial_r:.4f}")
    print(f"Regression R-squared: R² = {regression_r_squared:.4f}")

    print(f"\nOutput files written to: {output_dir.resolve()}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze FAA SDR AircraftTotalTime by PartCondition."
    )

    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help=f"FAA SDR calendar year to download. Default: {DEFAULT_YEAR}",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional local FAA SDR CSV file. If supplied, --year is not downloaded.",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory used for downloaded FAA CSV files. Default: data",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for tables and figures. Default: outputs",
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-download of the FAA SDR CSV file.",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip chart creation.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Run the full analysis pipeline."""
    args = parse_args(argv)

    try:
        df = load_sdr_data(
            input_path=args.input,
            year=args.year,
            data_dir=args.data_dir,
            refresh=args.refresh,
        )

        analysis = prepare_analysis_data(df)
        desc = descriptive_statistics(analysis)
        welch_results = welch_t_test(analysis)
        coef_table, point_biserial_r, regression_r_squared = regression_analysis(analysis)

        write_outputs(
            analysis=analysis,
            desc=desc,
            welch_results=welch_results,
            coef_table=coef_table,
            point_biserial_r=point_biserial_r,
            regression_r_squared=regression_r_squared,
            output_dir=args.output_dir,
            create_plots=not args.no_plots,
        )

        print_report(
            desc=desc,
            welch_results=welch_results,
            coef_table=coef_table,
            point_biserial_r=point_biserial_r,
            regression_r_squared=regression_r_squared,
            output_dir=args.output_dir,
        )

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
