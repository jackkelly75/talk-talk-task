import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def build_churn_dataset(
    customer_df,
    cease_df,
    calls_df,
    usage_df,
    churn_window_days=30,
    lookback_days=30
):
    # ensure dates are formatted corectly
    cease_df   = cease_df.copy()
    calls_df    = calls_df.copy()
    usage_df   = usage_df.copy()
    cease_df["cease_placed_date"]    = pd.to_datetime(cease_df["cease_placed_date"])
    calls_df["event_date"]            = pd.to_datetime(calls_df["event_date"])
    usage_df["calendar_date"]        = pd.to_datetime(usage_df["calendar_date"])
    customer_df["datevalue"]    = pd.to_datetime(customer_df["datevalue"])
    # Pre-compute first cease date per customer
    first_cease = (
        cease_df.groupby("unique_customer_identifier")["cease_placed_date"]
        .min()
        .reset_index()
        .rename(columns={"cease_placed_date": "first_cease_date"})
    )
    # Snapshot dates — exclude months where 30-day window is incomplete  #
    snapshot_dates = sorted(pd.to_datetime(customer_df["datevalue"].unique()))
    DATA_END_DATE = cease_df.cease_placed_date.max() - timedelta(days=churn_window_days) # 30 days before cease data maximum
    snapshot_dates = [s for s in snapshot_dates if s <= DATA_END_DATE]
    # Roll through each month as a snapshot
    all_snapshots = []
    for snapshot_date in snapshot_dates:
        print(snapshot_date)
        now = datetime.now()
        print(now.strftime("%Y-%m-%d %H:%M:%S"))
        window_end     = snapshot_date + pd.Timedelta(days=churn_window_days)
        lookback_start = snapshot_date - pd.Timedelta(days=lookback_days)
        recent_start   = snapshot_date - pd.Timedelta(days=30)   # last 30 days
        prior_start    = snapshot_date - pd.Timedelta(days=90)   # 30–90 days ago
        #active customers this month
        print("active customer features")
        already_churned = first_cease.loc[
            first_cease["first_cease_date"] <= snapshot_date,
            "unique_customer_identifier"
        ].unique()

        base = (
            customer_df[customer_df["datevalue"] == snapshot_date]
            .query("unique_customer_identifier not in @already_churned")
            [[
                "unique_customer_identifier",
                "contract_status", "contract_dd_cancels", "dd_cancel_60_day",
                "ooc_days", "technology", "speed", "line_speed",
                "sales_channel", "crm_package_name", "tenure_days"
            ]]
            .drop_duplicates("unique_customer_identifier")
            .copy()
        )
        base["snapshot_date"] = snapshot_date
        if base.empty:
            continue
        # Churn label
        churners = (
            first_cease[
                (first_cease["first_cease_date"] > snapshot_date) &
                (first_cease["first_cease_date"] <= window_end)
            ]
            [["unique_customer_identifier"]]
            .assign(churned=1)
        )
        # Customer-level features from customer_info 
        print("customer level features")
        now = datetime.now()
        print(now.strftime("%Y-%m-%d %H:%M:%S"))
        base["speed_deficit"]        = base["speed"] - base["line_speed"]
        base["speed_attainment_pct"] = base["line_speed"] / (base["speed"] + 1)
        base["poor_speed_flag"]      = (base["speed_attainment_pct"] < 0.7).astype(int)
        base["near_ooc_flag"]        = (base["ooc_days"].between(-30, 0)).astype(int)
        base["ooc_days_positive"]    = base["ooc_days"].clip(lower=0)
        base["days_until_ooc"]       = base["ooc_days"].clip(upper=0).abs()
        base["dd_cancel_rate"]       = base["contract_dd_cancels"] / (base["tenure_days"] / 30 + 1)
        base["tenure_band"]          = pd.cut(
            base["tenure_days"],
            bins=[0, 90, 180, 365, 730, 9999],
            labels=["0-3m", "3-6m", "6-12m", "1-2yr", "2yr+"]
        ).astype(str)
        # Call features
        print("call features")
        now = datetime.now()
        print(now.strftime("%Y-%m-%d %H:%M:%S"))
        calls_window  = calls_df[(calls_df["event_date"] > lookback_start) & (calls_df["event_date"] <= snapshot_date)]
        calls_recent  = calls_df[(calls_df["event_date"] > recent_start)   & (calls_df["event_date"] <= snapshot_date)]
        calls_prior   = calls_df[(calls_df["event_date"] > prior_start)    & (calls_df["event_date"] <= recent_start)]
        # Aggregate over full lookback window
        print("agg call features")
        now = datetime.now()
        print(now.strftime("%Y-%m-%d %H:%M:%S"))
        call_agg = (
            calls_window.groupby("unique_customer_identifier")
            .agg(
                total_calls          = ("event_date",          "count"),
                loyalty_calls        = ("call_type",           lambda x: (x == "Loyalty").sum()),
                csb_calls            = ("call_type",           lambda x: (x == "CS&B").sum()),
                ever_called_loyalty  = ("call_type",           lambda x: int("Loyalty" in x.values)),
                avg_talk_time        = ("talk_time_seconds",   "mean"),
                avg_hold_time        = ("hold_time_seconds",   "mean"),
                total_hold_time      = ("hold_time_seconds",   "sum"),
                days_since_last_call = ("event_date",          lambda x: (snapshot_date - x.max()).days),
                repeat_contact_flag  = ("event_date",          lambda x: int(x.nunique() > 1)),
            )
            .reset_index()
        )
        # Hold-to-talk ratio (join talk time back in for the ratio)
        print("Hold-to-talk features")
        now = datetime.now()
        print(now.strftime("%Y-%m-%d %H:%M:%S"))
        talk_totals = calls_window.groupby("unique_customer_identifier")["talk_time_seconds"].sum().rename("total_talk_time")
        hold_totals = calls_window.groupby("unique_customer_identifier")["hold_time_seconds"].sum().rename("total_hold_time_sum")
        call_agg = call_agg.merge(talk_totals, on="unique_customer_identifier", how="left")
        call_agg["hold_to_talk_ratio"] = call_agg["total_hold_time"] / (call_agg["total_talk_time"] + 1)
        call_agg.drop(columns=["total_talk_time"], inplace=True)
        # Recent vs prior call counts for velocity
        print("Recent vs prior call counts features")
        now = datetime.now()
        print(now.strftime("%Y-%m-%d %H:%M:%S"))
        calls_recent_cnt = calls_recent.groupby("unique_customer_identifier").size().rename("calls_last_30d")
        calls_prior_cnt  = calls_prior.groupby("unique_customer_identifier").size().rename("calls_prior_60d")
        call_agg = call_agg.merge(calls_recent_cnt, on="unique_customer_identifier", how="left")
        call_agg = call_agg.merge(calls_prior_cnt,  on="unique_customer_identifier", how="left")
        call_agg["calls_last_30d"]  = call_agg["calls_last_30d"].fillna(0)
        call_agg["calls_prior_60d"] = call_agg["calls_prior_60d"].fillna(0)
        call_agg["call_velocity"]   = call_agg["calls_last_30d"] - (call_agg["calls_prior_60d"] / 2)
        # ---- Usage features
        print("usage features")
        now = datetime.now()
        print(now.strftime("%Y-%m-%d %H:%M:%S"))
        usage_window = usage_df[(usage_df["calendar_date"] > lookback_start) & (usage_df["calendar_date"] <= snapshot_date)]
        usage_recent = usage_df[(usage_df["calendar_date"] > recent_start)   & (usage_df["calendar_date"] <= snapshot_date)]
        usage_prior  = usage_df[(usage_df["calendar_date"] > prior_start)    & (usage_df["calendar_date"] <= recent_start)]

        usage_agg = (
            usage_window.groupby("unique_customer_identifier")
            .agg(
                avg_download_mbs   = ("usage_download_mbs", "mean"),
                avg_upload_mbs     = ("usage_upload_mbs",   "mean"),
                total_download_mbs = ("usage_download_mbs", "sum"),
                total_upload_mbs   = ("usage_upload_mbs",   "sum"),
            )
            .reset_index()
        )
        # Download trend: % change recent vs prior period
        print("Download trend features")
        now = datetime.now()
        print(now.strftime("%Y-%m-%d %H:%M:%S"))
        recent_dl = usage_recent.groupby("unique_customer_identifier")["usage_download_mbs"].mean().rename("recent_avg_dl")
        prior_dl  = usage_prior.groupby("unique_customer_identifier")["usage_download_mbs"].mean().rename("prior_avg_dl")
        usage_agg = usage_agg.merge(recent_dl, on="unique_customer_identifier", how="left")
        usage_agg = usage_agg.merge(prior_dl,  on="unique_customer_identifier", how="left")
        usage_agg["download_trend_pct"] = (
            (usage_agg["recent_avg_dl"] - usage_agg["prior_avg_dl"]) / (usage_agg["prior_avg_dl"] + 1)
        )
        usage_agg.drop(columns=["recent_avg_dl", "prior_avg_dl"], inplace=True)
        # Days since last usage
        print("Days since last usage features")
        now = datetime.now()
        print(now.strftime("%Y-%m-%d %H:%M:%S"))
        last_usage = (
            usage_df[usage_df["calendar_date"] <= snapshot_date]
            .groupby("unique_customer_identifier")["calendar_date"]
            .max()
            .reset_index()
            .rename(columns={"calendar_date": "last_usage_date"})
        )
        last_usage["days_since_last_usage"] = (snapshot_date - last_usage["last_usage_date"]).dt.days
        last_usage.drop(columns=["last_usage_date"], inplace=True)
        # 0 usage days in last 30 days
        print("0 usage features")
        now = datetime.now()
        print(now.strftime("%Y-%m-%d %H:%M:%S"))
        zero_usage = (
            usage_recent
            .assign(zero_flag=lambda x: (x["usage_download_mbs"] == 0).astype(int))
            .groupby("unique_customer_identifier")["zero_flag"]
            .sum()
            .rename("zero_usage_days_30d")
            .reset_index()
        )
        usage_agg = usage_agg.merge(last_usage, on="unique_customer_identifier", how="left")
        usage_agg = usage_agg.merge(zero_usage, on="unique_customer_identifier", how="left")
        # joinall features
        snapshot = (
            base
            .merge(call_agg,   on="unique_customer_identifier", how="left")
            .merge(usage_agg,  on="unique_customer_identifier", how="left")
            .merge(churners,   on="unique_customer_identifier", how="left")
        )
        snapshot["churned"] = snapshot["churned"].fillna(0).astype(int)
        all_snapshots.append(snapshot)

    # Combine, fill nulls, add cross-snapshot rolling features         #
    print("combining data")
    now = datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    model_df = pd.concat(all_snapshots, ignore_index=True)
    model_df = model_df.sort_values(["unique_customer_identifier", "snapshot_date"]).reset_index(drop=True)

    # Fill nulls for customers with no calls or usage activity
    call_fill_cols  = ["total_calls", "loyalty_calls", "csb_calls", "ever_called_loyalty",
                       "avg_talk_time", "avg_hold_time", "total_hold_time", "days_since_last_call",
                       "repeat_contact_flag", "hold_to_talk_ratio", "calls_last_30d",
                       "calls_prior_60d", "call_velocity"]
    usage_fill_cols = ["avg_download_mbs", "avg_upload_mbs", "total_download_mbs",
                       "total_upload_mbs", "download_trend_pct", "days_since_last_usage",
                       "zero_usage_days_30d"]

    model_df[call_fill_cols]  = model_df[call_fill_cols].fillna(0)
    model_df[usage_fill_cols] = model_df[usage_fill_cols].fillna(0)

    # ---- Cross-snapshot rolling features (require full panel) ---------- #
    grp = model_df.groupby("unique_customer_identifier")
    # Rolling 3-month call and loyalty trends
    model_df["calls_3m_rolling"]       = grp["total_calls"].transform(lambda x: x.rolling(3, min_periods=1).sum())
    model_df["loyalty_call_ever_3m"]   = grp["ever_called_loyalty"].transform(lambda x: x.rolling(3, min_periods=1).max())
    #how many consecutive months they've had zero usage days flagged
    model_df["consecutive_zero_months"] = (
        grp["zero_usage_days_30d"]
        .transform(lambda x: x.gt(0).groupby((~x.gt(0)).cumsum()).cumsum())
    )
    #usage trend direction over 3 months — sustained decline is a strong signal
    model_df["download_trend_3m"] = grp["avg_download_mbs"].transform(
        lambda x: x.rolling(3, min_periods=2).apply(lambda w: np.polyfit(range(len(w)), w, 1)[0], raw=True)
    )
    return model_df


