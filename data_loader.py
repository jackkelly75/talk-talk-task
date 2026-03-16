import pandas as pd
import duckdb
from pathlib import Path

def load_all_data(data_dir, verbose=False):
    data_dir = Path(data_dir)
    # --- Cease data 
    cease_df = pd.read_csv(
        data_dir / "cease.csv",
        parse_dates=["cease_placed_date", "cease_completed_date"]
    )
    print(f"Cease data: {cease_df.shape}")
    print(cease_df.head())
    # --- Customer info
    customer_df = pd.read_parquet(data_dir / "customer_info.parquet")
    print(f"Customer info: {customer_df.shape}")
    # --- Calls data 
    calls_df = pd.read_csv(
        data_dir / "calls.csv",
        parse_dates=["event_date"]
    )
    print(f"Calls data: {calls_df.shape}")
    # --- Usage data (DuckDB for efficiency) 
    con = duckdb.connect()
    usage_df = con.execute(f"""
        SELECT 
            unique_customer_identifier,
            calendar_date,
            usage_download_mbs,
            usage_upload_mbs
        FROM read_parquet('{data_dir}/usage.parquet')
    """).df()
    con.close()

    # Clean numeric fields
    usage_df["usage_download_mbs"] = pd.to_numeric(usage_df["usage_download_mbs"], errors="coerce")
    usage_df["usage_upload_mbs"] = pd.to_numeric(usage_df["usage_upload_mbs"], errors="coerce")

    print(f"Usage data: {usage_df.shape}")

    return {
        "cease": cease_df,
        "customer": customer_df,
        "calls": calls_df,
        "usage": usage_df
    }