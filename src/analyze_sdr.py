from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    args = parser.parse_args()

    year = args.year

    base_dir = Path(__file__).resolve().parents[1]
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    tables_dir = base_dir / "outputs" / "tables"
    figures_dir = base_dir / "outputs" / "figures"

    for folder in [raw_dir, processed_dir, tables_dir, figures_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    url = f"https://external.apic4e.faa.gov/sdrs/retrieve/SDR-{year}.csv"
    raw_file = raw_dir / f"SDR-{year}.csv"

    print(f"Using FAA SDR data year: {year}")
    print(f"Source URL: {url}")
    print(f"Local raw file: {raw_file}")

    # Always redownload so old cached data does not silently affect the results.
    df = pd.read_csv(url, low_memory=False)
    df.to_csv(raw_file, index=False)

    required_columns = ["PartCondition", "AircraftTotalTime"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df["PartCondition"] = df["PartCondition"].astype(str).str.upper().str.strip()
    df["AircraftTotalTime"] = pd.to_numeric(df["AircraftTotalTime"], errors="coerce")

    analysis = df.loc[
        df["PartCondition"].isin(["CRACKED", "CORRODED"]),
        ["PartCondition", "AircraftTotalTime"]
    ].dropna(subset=["AircraftTotalTime"]).copy()

    analysis = analysis[analysis["AircraftTotalTime"] >= 0].copy()

    processed_file = processed_dir / f"analysis_subset_{year}.csv"
    analysis.to_csv(processed_file, index=False)

    cracked = analysis.loc[
        analysis["PartCondition"] == "CRACKED",
        "AircraftTotalTime"
    ]

    corroded = analysis.loc[
        analysis["PartCondition"] == "CORRODED",
        "AircraftTotalTime"
    ]

    cracked_n = len(cracked)
    corroded_n = len(corroded)
    total_n = len(analysis)

    cracked_mean = cracked.mean()
    corroded_mean = corroded.mean()
    mean_difference = cracked_mean - corroded_mean

    cracked_var = cracked.var(ddof=1)
    corroded_var = corroded.var(ddof=1)

    welch_se = np.sqrt((cracked_var / cracked_n) + (corroded_var / corroded_n))

    welch_df = (
        ((cracked_var / cracked_n) + (corroded_var / corroded_n)) ** 2
        / (
            ((cracked_var / cracked_n) ** 2 / (cracked_n - 1))
            + ((corroded_var / corroded_n) ** 2 / (corroded_n - 1))
        )
    )

    t_statistic = mean_difference / welch_se
    p_value = 2 * stats.t.sf(abs(t_statistic), df=welch_df)

    t_critical = stats.t.ppf(0.975, df=welch_df)
    ci_95_lower = mean_difference - t_critical * welch_se
    ci_95_upper = mean_difference + t_critical * welch_se

    effect_r_from_t = np.sqrt((t_statistic ** 2) / ((t_statistic ** 2) + welch_df))
    effect_r_squared_from_t = effect_r_from_t ** 2

    analysis["Cracked"] = np.where(analysis["PartCondition"] == "CRACKED", 1, 0)

    point_biserial_r = analysis["AircraftTotalTime"].corr(analysis["Cracked"])

    model = smf.ols("AircraftTotalTime ~ Cracked", data=analysis).fit(cov_type="HC3")

    regression_intercept = model.params["Intercept"]
    regression_slope_cracked = model.params["Cracked"]
    regression_slope_se = model.bse["Cracked"]
    regression_r_squared = model.rsquared

    desc = (
        analysis.groupby("PartCondition")["AircraftTotalTime"]
        .agg(n="count", mean="mean", median="median", std="std", min="min", max="max")
        .reset_index()
    )

    desc["percent_of_subset"] = desc["n"] / total_n * 100
    desc = desc[
        ["PartCondition", "n", "percent_of_subset", "mean", "median", "std", "min", "max"]
    ]

    desc.to_csv(tables_dir / "descriptive_statistics.csv", index=False)

    summary = pd.DataFrame(
        {
            "statistic": [
                "total_n",
                "cracked_n",
                "corroded_n",
                "cracked_mean",
                "corroded_mean",
                "mean_difference_cracked_minus_corroded",
                "welch_standard_error",
                "welch_degrees_of_freedom",
                "t_statistic",
                "p_value",
                "ci_95_lower",
                "ci_95_upper",
                "effect_r_from_t",
                "effect_r_squared_from_t",
                "point_biserial_r",
                "regression_intercept",
                "regression_slope_cracked",
                "regression_slope_se",
                "regression_r_squared",
            ],
            "value": [
                total_n,
                cracked_n,
                corroded_n,
                cracked_mean,
                corroded_mean,
                mean_difference,
                welch_se,
                welch_df,
                t_statistic,
                p_value,
                ci_95_lower,
                ci_95_upper,
                effect_r_from_t,
                effect_r_squared_from_t,
                point_biserial_r,
                regression_intercept,
                regression_slope_cracked,
                regression_slope_se,
                regression_r_squared,
            ],
        }
    )

    summary.to_csv(tables_dir / "summary_results.csv", index=False)

    order = ["CORRODED", "CRACKED"]

    plt.figure()
    analysis["PartCondition"].value_counts().reindex(order).plot(kind="bar")
    plt.xlabel("Part condition")
    plt.ylabel("Number of reports")
    plt.title("Report Counts by PartCondition")
    plt.tight_layout()
    plt.savefig(figures_dir / "report_counts_by_part_condition.png", dpi=300)
    plt.close()

    plt.figure()
    analysis.groupby("PartCondition")["AircraftTotalTime"].mean().reindex(order).plot(kind="bar")
    plt.xlabel("Part condition")
    plt.ylabel("Mean aircraft total time, hours")
    plt.title("Mean AircraftTotalTime by PartCondition")
    plt.tight_layout()
    plt.savefig(figures_dir / "mean_aircraft_total_time_by_part_condition.png", dpi=300)
    plt.close()

    print("\nAnalysis complete.")
    print(f"Total analytic subset: {total_n:,}")
    print(f"CRACKED n: {cracked_n:,}")
    print(f"CORRODED n: {corroded_n:,}")
    print(f"CRACKED mean: {cracked_mean:,.2f}")
    print(f"CORRODED mean: {corroded_mean:,.2f}")
    print(f"Mean difference: {mean_difference:,.2f}")
    print(f"t({welch_df:,.2f}) = {t_statistic:,.2f}")
    print(f"95% CI: [{ci_95_lower:,.2f}, {ci_95_upper:,.2f}]")
    print(f"Regression slope SE: {regression_slope_se:,.2f}")
    print(f"R-squared: {regression_r_squared:.3f}")


if __name__ == "__main__":
    main()
