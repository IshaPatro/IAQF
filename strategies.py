import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import time

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@st.cache_data
def load_all():
    btc_usd = pd.read_csv(os.path.join(DATA_DIR, "BTC_USD_coinbase.csv"), parse_dates=["timestamp"])
    btc_usdt = pd.read_csv(os.path.join(DATA_DIR, "BTC_USDT_coinbase.csv"), parse_dates=["timestamp"])
    usdt_usd = pd.read_csv(os.path.join(DATA_DIR, "USDT_USD_coinbase.csv"), parse_dates=["timestamp"])

    btc_usd = btc_usd.rename(columns={"close": "btc_usd", "volume": "vol_usd"})
    btc_usdt = btc_usdt.rename(columns={"close": "btc_usdt", "volume": "vol_usdt"})
    usdt_usd = usdt_usd.rename(columns={"close": "usdt_usd"})

    merged = btc_usd[["timestamp", "btc_usd", "vol_usd"]].merge(
        btc_usdt[["timestamp", "btc_usdt", "vol_usdt"]], on="timestamp", how="inner"
    ).merge(
        usdt_usd[["timestamp", "usdt_usd"]], on="timestamp", how="inner"
    ).sort_values("timestamp").reset_index(drop=True)

    return merged


@st.cache_data
def downsample(df, freq="5min"):
    resampled = df.set_index("timestamp").resample(freq).agg({
        "btc_usd": "last",
        "btc_usdt": "last",
        "usdt_usd": "last",
        "vol_usd": "sum",
        "vol_usdt": "sum",
    }).dropna().reset_index()
    return resampled


def _chart(fig, title, yaxis_title, height=400):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, family="Inter")),
        yaxis_title=yaxis_title,
        template="plotly_dark",
        height=height,
        margin=dict(l=50, r=20, t=45, b=35),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    return fig


def _metrics_row(trades_df):
    if trades_df.empty:
        st.warning("No trades generated.")
        return

    total_pnl = trades_df["pnl_pct"].sum()
    win_rate = (trades_df["pnl_pct"] > 0).mean() * 100
    num_trades = len(trades_df)
    cum = trades_df["pnl_pct"].cumsum()
    running_max = cum.cummax()
    drawdown = cum - running_max
    max_dd = drawdown.min()
    avg_pnl = trades_df["pnl_pct"].mean()
    std_pnl = trades_df["pnl_pct"].std()
    sharpe = (avg_pnl / std_pnl) * np.sqrt(num_trades) if std_pnl > 0 else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total PnL", f"{total_pnl:.2f}%")
    with c2:
        st.metric("Sharpe", f"{sharpe:.2f}")
    with c3:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with c4:
        st.metric("Max Drawdown", f"{max_dd:.2f}%")
    with c5:
        st.metric("# Trades", f"{num_trades}")


def _price_chart_with_signals(df_plot, trades_df, price_col, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_plot["timestamp"], y=df_plot[price_col],
        line=dict(color="rgba(255,255,255,0.5)", width=1),
        name="Price",
    ))

    if not trades_df.empty:
        buys = trades_df[trades_df["side"].str.contains("Long")]
        sells = trades_df[trades_df["side"].str.contains("Short")]

        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys["entry_time"], y=buys["entry_price"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=12, color="#4ECDC4", line=dict(width=1, color="white")),
                name="Buy",
            ))
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells["entry_time"], y=sells["entry_price"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=12, color="#FF6B6B", line=dict(width=1, color="white")),
                name="Sell",
            ))

        for _, t in trades_df.iterrows():
            color = "#4ECDC4" if t["pnl_pct"] > 0 else "#FF6B6B"
            fig.add_trace(go.Scatter(
                x=[t["entry_time"], t["exit_time"]],
                y=[t["entry_price"], t["exit_price"]],
                mode="lines",
                line=dict(color=color, width=0.8, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            ))

    _chart(fig, title, "Price (USD)")
    return fig


def _equity_curve(trades_df):
    if trades_df.empty:
        return
    cum = trades_df["pnl_pct"].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trades_df["exit_time"], y=cum,
        line=dict(color="#4ECDC4", width=1.5),
        fill="tozeroy", fillcolor="rgba(78,205,196,0.1)",
        name="Cumulative PnL",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
    _chart(fig, "Equity Curve (Cumulative PnL %)", "PnL (%)")
    st.plotly_chart(fig, use_container_width=True)


def _order_book(trades_df):
    if trades_df.empty:
        return
    display = trades_df.copy()
    display["entry_time"] = display["entry_time"].dt.strftime("%m/%d %H:%M")
    display["exit_time"] = display["exit_time"].dt.strftime("%m/%d %H:%M")
    for col in ["entry_price", "exit_price"]:
        display[col] = display[col].map(lambda x: f"${x:,.2f}" if x > 10 else f"${x:.6f}")
    display["pnl_pct"] = display["pnl_pct"].map(lambda x: f"{x:+.4f}%")
    display.columns = ["Entry", "Exit", "Side", "Entry Price", "Exit Price", "PnL (%)"]

    def highlight_row(row):
        is_long = "Long" in str(row["Side"])
        color = "rgba(78, 205, 196, 0.15)" if is_long else "rgba(255, 107, 107, 0.15)"
        return [f"background-color: {color}"] * len(row)

    st.dataframe(display.style.apply(highlight_row, axis=1), use_container_width=True, hide_index=True)


def _simulate_price(df_plot, trades_df, price_col, title, speed=0.03):
    chart_placeholder = st.empty()
    n = len(df_plot)
    step = max(1, n // 80)

    for end in range(step, n + 1, step):
        chunk = df_plot.iloc[:end]
        cutoff = chunk["timestamp"].iloc[-1]
        if hasattr(cutoff, "tzinfo") and cutoff.tzinfo is not None:
            cutoff = cutoff.tz_localize(None)
        visible_trades = trades_df[trades_df["entry_time"] <= cutoff] if not trades_df.empty else trades_df
        fig = _price_chart_with_signals(chunk, visible_trades, price_col, title)
        chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"sim_{end}")
        time.sleep(speed)

    fig = _price_chart_with_signals(df_plot, trades_df, price_col, title)
    chart_placeholder.plotly_chart(fig, use_container_width=True, key="sim_final")


def backtest_basis_arb(df):
    basis = np.log(df["btc_usd"].values) - np.log(df["btc_usdt"].values)
    rolling_mean = pd.Series(basis).rolling(120, min_periods=60).mean().values
    rolling_std = pd.Series(basis).rolling(120, min_periods=60).std().values

    trades = []
    position = None
    ts = df["timestamp"].values
    prices = df["btc_usd"].values

    for i in range(120, len(df)):
        if rolling_std[i] == 0 or np.isnan(rolling_std[i]):
            continue
        z = (basis[i] - rolling_mean[i]) / rolling_std[i]

        if position is None:
            if z > 2.0:
                position = {"side": "Short Basis", "entry_idx": i, "entry_price": prices[i]}
            elif z < -2.0:
                position = {"side": "Long Basis", "entry_idx": i, "entry_price": prices[i]}
        else:
            exit_signal = abs(z) < 0.5 or abs(z) > 4.0
            hold_time = i - position["entry_idx"]
            if exit_signal or hold_time > 360:
                ep = prices[i]
                pnl = -((ep / position["entry_price"]) - 1) * 100 if position["side"] == "Short Basis" else ((ep / position["entry_price"]) - 1) * 100
                trades.append({
                    "entry_time": pd.Timestamp(ts[position["entry_idx"]]),
                    "exit_time": pd.Timestamp(ts[i]),
                    "side": position["side"],
                    "entry_price": position["entry_price"],
                    "exit_price": ep,
                    "pnl_pct": pnl,
                })
                position = None

    return pd.DataFrame(trades)


def render():
    df = load_all()
    df_5m = downsample(df, "5min")

    st.subheader("Cross-Currency Arbitrage Strategy: Basis Mean-Reversion")
    st.markdown(
        """
        Exploits the decoupling between BTC/USD and BTC/USDT during stress. When the basis 
        deviates beyond a Z-score threshold, we enter a convergence trade expecting mean-reversion.
        """
    )
    st.latex(r"Z_t = \frac{Basis_t - \mu_{120}}{\sigma_{120}}, \quad Basis_t = \ln(P_{BTC/USD}) - \ln(P_{BTC/USDT})")
    st.markdown(
        """
        - **Entry:** |Z| > 2.0 → Long cheap leg, Short expensive leg
        - **Exit:** |Z| < 0.5 (convergence) or |Z| > 4.0 (stop-loss)
        - **Max Hold:** 6 hours
        """
    )

    if st.button("▶ Run Backtest", key="basis_arb"):
        with st.spinner("Running basis arbitrage backtest..."):
            trades = backtest_basis_arb(df)

        _simulate_price(df_5m, trades, "btc_usd", "BTC/USD — Basis Arb Signals")
        
        st.divider()
        st.subheader("Performance Metrics")
        _metrics_row(trades)
        
        st.subheader("Equity Curve")
        _equity_curve(trades)
        
        st.subheader("Order Book")
        _order_book(trades)
