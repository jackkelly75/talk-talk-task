import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
def generate_churn_dashboard(customer_df, cease_df):
    all_customers = (
        customer_df
        .sort_values("datevalue")
        .drop_duplicates(subset="unique_customer_identifier", keep="last")
        .copy()
    )
    date_min = customer_df["datevalue"].min()
    date_max = customer_df["datevalue"].max()
    print(all_customers["contract_status"].value_counts())
    print(all_customers["technology"].value_counts())
    print(all_customers["sales_channel"].value_counts())
    # Churn flag on the deduplicated customer table
    ceased_ids = set(cease_df["unique_customer_identifier"])
    all_customers["churned"] = all_customers["unique_customer_identifier"].isin(ceased_ids).astype(int)
    # --- Dashboard styling
    PALETTE = sns.color_palette("Blues_r", 10)
    BG = "#f7f9fc"
    ACCENT = "#1a6eb5"
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "sans-serif", "axes.titleweight": "bold",
    })
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"Telecoms Churn Dashboard  ({date_min:%b %Y} – {date_max:%b %Y})",
        fontsize=20, fontweight="bold", y=1.01,
    )
    axes = axes.flatten()
    # 1. Churn reason breakdown (all churners — no date filter needed)
    reason_counts = (
        cease_df["reason_description_insight"]
        .value_counts(normalize=True)
        .round(3)
        .sort_values()
    )
    axes[0].barh(reason_counts.index, reason_counts.values, color=PALETTE[:len(reason_counts)])
    axes[0].xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axes[0].set_title("Cease Reason Breakdown")
    axes[0].set_xlabel("Share of Churners")
    for i, v in enumerate(reason_counts.values):
        axes[0].text(v + 0.002, i, f"{v:.1%}", va="center", fontsize=8)
    # 2. Contract status distribution (all unique customers)
    contract_counts = all_customers["contract_status"].value_counts()
    axes[1].bar(contract_counts.index, contract_counts.values,
                color=sns.color_palette("Blues_r", len(contract_counts)))
    axes[1].set_title("Contract Status Distribution\n(all unique customers)")
    axes[1].set_ylabel("Customer Count")
    axes[1].tick_params(axis="x", rotation=20)
    for i, v in enumerate(contract_counts.values):
        axes[1].text(i, v + contract_counts.max() * 0.01, f"{v:,}", ha="center", fontsize=8)
    # 3. Technology split (all unique customers)
    tech_counts = all_customers["technology"].value_counts()
    wedge_props = {"linewidth": 2, "edgecolor": BG}
    axes[2].pie(
        tech_counts.values,
        labels=tech_counts.index,
        autopct="%1.1f%%",
        colors=sns.color_palette("Blues_r", len(tech_counts)),
        wedgeprops=wedge_props,
        startangle=140,
    )
    axes[2].set_title("Technology Split\n(all unique customers)")
    # 4. Sales channel (all unique customers)
    channel_counts = all_customers["sales_channel"].value_counts().sort_values()
    axes[3].barh(channel_counts.index, channel_counts.values,
                 color=sns.color_palette("Blues", len(channel_counts)))
    axes[3].set_title("Sales Channel\n(all unique customers)")
    axes[3].set_xlabel("Customer Count")
    for i, v in enumerate(channel_counts.values):
        axes[3].text(v + channel_counts.max() * 0.01, i, f"{v:,}", va="center", fontsize=8)
    # 5. Churn rate by contract status (all unique customers)
    churn_by_status = (
        all_customers.groupby("contract_status")["churned"]
        .mean()
        .sort_values(ascending=False)
    )
    bars = axes[4].bar(
        churn_by_status.index, churn_by_status.values,
        color=[ACCENT if v == churn_by_status.max() else "#a8c8e8" for v in churn_by_status.values],
    )
    axes[4].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axes[4].set_title("Churn Rate by Contract Status\n(all unique customers)")
    axes[4].set_ylabel("Churn Rate")
    axes[4].tick_params(axis="x", rotation=20)
    for bar, v in zip(bars, churn_by_status.values):
        axes[4].text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                     f"{v:.1%}", ha="center", fontsize=8)
    axes[5].set_visible(False)
    fig.tight_layout()
    plt.show()

    return all_customers