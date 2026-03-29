import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Seasonal Study", layout="wide")

st.title("Seasonal Study App")

# Sidebar controls
st.sidebar.header("Settings")

symbol = st.sidebar.text_input("Symbol", value="AUDCHF=X")
start_year = st.sidebar.number_input("Start Year", min_value=1900, max_value=2100, value=2010, step=1)
end_year = st.sidebar.number_input("End Year", min_value=1900, max_value=2100, value=2025, step=1)
slice_start = st.sidebar.text_input("Slice Start (MM-DD)", value="08-01")
slice_end = st.sidebar.text_input("Slice End (MM-DD)", value="09-01")

run = st.sidebar.button("Run Study")

def run_study(symbol: str, start_year: int, end_year: int, slice_start: str, slice_end: str):
    # Download enough data to cover the whole range
    download_start = f"{start_year}-01-01"
    download_end = f"{end_year}-12-31"

    df = yf.download(symbol, start=download_start, end=download_end, auto_adjust=True, progress=False)

    if df.empty:
        st.error("No data returned. Check the symbol and date range.")
        return

    # Use Close column safely
    if "Close" not in df.columns:
        st.error("Downloaded data does not contain a Close column.")
        return

    

    prices = df["Close"]
    
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    
    prices = prices.astype(float).dropna()
    
    yearly_returns = []
    cumulative_returns = []
    valid_years = []
    
    for year in range(start_year, end_year + 1):
    
        try:
    
            start = pd.to_datetime(f"{year}-{slice_start}")
            end = pd.to_datetime(f"{year}-{slice_end}")
    
            if start > prices.index[-1] or end > prices.index[-1]:
                continue
    
            start_idx = prices.index.get_indexer([start], method="nearest")[0]
            end_idx = prices.index.get_indexer([end], method="nearest")[0]
    
            if end_idx <= start_idx:
                continue
    
            start_price = float(prices.iloc[start_idx])
            end_price = float(prices.iloc[end_idx])
    
            ret = (end_price - start_price) / start_price * 100
    
            yearly_returns.append(ret)
            valid_years.append(year)
    
            if not cumulative_returns:
                cumulative_returns.append(1 + ret / 100)
            else:
                cumulative_returns.append(
                    cumulative_returns[-1] * (1 + ret / 100)
                )
    
        except Exception:
            continue
    if not yearly_returns:
        st.warning("No valid yearly returns found for that setup.")
        return

    # Step 4: Stats
    returns_series = pd.Series(yearly_returns)
    wins = (returns_series > 0).sum()
    losses = (returns_series <= 0).sum()
    avg_return = returns_series.mean()
    median_return = returns_series.median()
    win_rate = wins / len(returns_series) * 100
    total_return = cumulative_returns[-1] - 1 if cumulative_returns else 0

    # Step 5: Average full-year path
    full_year_paths = []
    slice_start_offsets = []
    slice_end_offsets = []
    max_len = 0

    for year in range(start_year, end_year + 1):
        try:
            ystart = pd.to_datetime(f"{year}-01-01")
            yend = pd.to_datetime(f"{year}-12-31")

            if yend > prices.index[-1]:
                continue

            year_data = prices.loc[ystart:yend].copy()
            if len(year_data) < 200:
                continue

            norm = year_data / year_data.iloc[0]
            norm.index = range(len(norm))
            full_year_paths.append(norm)
            max_len = max(max_len, len(norm))

            s_idx = prices.index.get_indexer([pd.Timestamp(f"{year}-{slice_start}")], method="nearest")[0]
            e_idx = prices.index.get_indexer([pd.Timestamp(f"{year}-{slice_end}")], method="nearest")[0]
            base_idx = prices.index.get_indexer([ystart], method="nearest")[0]

            slice_start_offsets.append(s_idx - base_idx)
            slice_end_offsets.append(e_idx - base_idx)
        except Exception:
            continue

    # Step 6: Average and clean
    full_year_avg_path = None
    slice_start_avg = None
    slice_end_avg = None

    if full_year_paths:
        padded = [p.reindex(range(max_len)) for p in full_year_paths]
        matrix = pd.concat(padded, axis=1)

        valid_counts = matrix.count(axis=1)
        min_years = 3
        full_year_avg_path = matrix[valid_counts >= min_years].mean(axis=1, skipna=True)

        if slice_start_offsets and slice_end_offsets:
            slice_start_avg = int(np.round(np.mean(slice_start_offsets)))
            slice_end_avg = int(np.round(np.mean(slice_end_offsets)))

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Years", len(yearly_returns))
    c2.metric("Win Rate", f"{win_rate:.2f}%")
    c3.metric("Average Return", f"{avg_return:.2f}%")
    c4.metric("Cumulative Return", f"{total_return * 100:.2f}%")

    # Returns table
    st.subheader("Returns by Year")
    returns_df = pd.DataFrame({
        "Year": valid_years,
        "Return %": yearly_returns
    })
    st.dataframe(returns_df, use_container_width=True)

    # Plot
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # [0,0] Avg path
    if full_year_avg_path is not None:
        axes[0, 0].plot(full_year_avg_path.index, full_year_avg_path.values, label="Avg Yearly Path", color="blue")
        axes[0, 0].axhline(1.0, linestyle="--", color="gray")
        if slice_start_avg is not None:
            axes[0, 0].axvline(slice_start_avg, linestyle="--", color="red", label="Slice Start")
        if slice_end_avg is not None:
            axes[0, 0].axvline(slice_end_avg, linestyle="--", color="green", label="Slice End")
        axes[0, 0].set_title("Average Normalized Price Path (Full Year)")
        axes[0, 0].set_ylabel("Normalized Price")
        axes[0, 0].set_xlabel("Trading Days Into Year")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    else:
        axes[0, 0].text(0.1, 0.5, "No valid path data", fontsize=12)
        axes[0, 0].set_axis_off()

    # [0,1] Returns by year
    axes[0, 1].bar(
        range(len(yearly_returns)),
        yearly_returns,
        color=["green" if r > 0 else "red" for r in yearly_returns]
    )
    axes[0, 1].set_title("Pattern Returns by Year")
    axes[0, 1].axhline(0, color="black")
    axes[0, 1].set_xticks(range(len(valid_years)))
    axes[0, 1].set_xticklabels(valid_years, rotation=45)
    axes[0, 1].grid(True)

    # [1,0] Cumulative
    axes[1, 0].plot(range(len(cumulative_returns)), cumulative_returns, marker="o")
    axes[1, 0].set_title("Cumulative Return")
    axes[1, 0].set_ylabel("Growth of $1")
    axes[1, 0].grid(True)

    # [1,1] Pie
    axes[1, 1].pie(
        [wins, losses],
        labels=["Wins", "Losses"],
        autopct="%1.1f%%",
        colors=["green", "red"]
    )
    axes[1, 1].set_title("Win/Loss Distribution")

    # [2,0] Stats box
    axes[2, 0].axis("off")
    summary = f"""
Slice: {slice_start} to {slice_end}
Years: {start_year}–{end_year}
---------------------------
Total Years: {len(yearly_returns)}
Winning Trades: {wins}
Losing Trades: {losses}
Win Rate: {win_rate:.2f}%
Average Return: {avg_return:.2f}%
Median Return: {median_return:.2f}%
Cumulative Return: {total_return * 100:.2f}%
"""
    axes[2, 0].text(
        0, 0.5, summary,
        fontsize=12,
        verticalalignment="center",
        fontfamily="monospace"
    )

    # [2,1] Empty
    axes[2, 1].axis("off")

    plt.suptitle(f"Seasonal Study: {symbol} from {slice_start} to {slice_end}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    st.pyplot(fig)

if run:
    run_study(symbol, int(start_year), int(end_year), slice_start, slice_end)
else:
    st.info("Set your parameters in the sidebar and click 'Run Study'.")
