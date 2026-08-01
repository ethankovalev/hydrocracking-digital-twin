"""
STEP 0: clean the raw SCADA telemetry into a validated parquet file.

Run this first - everything downstream reads its output.
"""

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def csv_to_parquet():
    """Only needed if you are starting from a raw CSV."""
    if not config.RAW_CSV.exists():
        return
    pl.read_csv(config.RAW_CSV).write_parquet(config.PARQUET)
    print(f"[ok] converted CSV -> {config.PARQUET}")


def clean():
    if not config.PARQUET.exists():
        raise FileNotFoundError(
            f"No telemetry at {config.PARQUET}\n"
            "Put reactor_data.parquet (or sample_data.csv) in the data/ folder."
        )

    df = (
        pl.scan_parquet(config.PARQUET)

        .filter(
            (pl.col("quality_issue") == "GOOD")
            & (pl.col("missing_flag") == 0)
            & (pl.col("operating_mode") == "NORMAL")
            & (pl.col("quality_score") == 1)
        )

        .with_columns([
            pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S"),

            # Celsius -> Kelvin
            (pl.col("reactor_inlet_temp_C") + 273.15).alias("temp_inlet_K"),
            (pl.col("heater_outlet_temp_C") + 273.15).alias("temp_outlet_K"),
            (pl.col("feed_temp_C") + 273.15).alias("feed_temp_K"),

            # barg -> Pa. The x1e5 MUST come first. Adding 101325 to a value
            # in bar makes a 150 barg reactor read as ~1 atm, silently
            # corrupting every downstream calculation.
            (pl.col("reactor_pressure_barg") * 1e5 + 101325).alias("pressure_Pa"),
            (pl.col("reactor_delta_p_bar") * 1e5).alias("delta_p_Pa"),
        ])

        # scan_parquet gives no ordering guarantee and this is a time series
        .sort("timestamp")

        .select([
            "timestamp", "minute_index",
            "wax_feed_tph", "feed_density_proxy",
            "fresh_h2_flow_Nm3h", "recycle_gas_flow_Nm3h",
            "quench1_flow_tph", "quench2_flow_tph",
            "temp_inlet_K", "temp_outlet_K", "feed_temp_K",
            "pressure_Pa", "delta_p_Pa",
            "catalyst_age_days",
            "conversion_pct", "naphtha_yield_pct",
        ])
        .collect()
    )

    df.write_parquet(config.CLEAN_PARQUET)

    print(f"[ok] {len(df)} clean rows -> {config.CLEAN_PARQUET}")
    print(f"     temperature : {df['temp_inlet_K'].min():.1f} - "
          f"{df['temp_inlet_K'].max():.1f} K")
    print(f"     pressure    : {df['pressure_Pa'].min()/1e5:.1f} - "
          f"{df['pressure_Pa'].max()/1e5:.1f} bar")
    print(f"     conversion  : {df['conversion_pct'].min():.1f} - "
          f"{df['conversion_pct'].max():.1f} %")
    return df


def main():
    csv_to_parquet()
    return clean()


if __name__ == "__main__":
    main()
