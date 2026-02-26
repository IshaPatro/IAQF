import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
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


def _order_book(trades_df):
    if trades_df.empty:
        return
    display = trades_df.copy()
    display["entry_time"] = display["entry_time"].dt.strftime("%m/%d %H:%M")
    display["exit_time"] = display["exit_time"].dt.strftime("%m/%d %H:%M")
    for col in ["entry_price", "exit_price"]:
        display[col] = display[col].map(lambda x: f"${x:,.2f}" if x > 10 else f"${x:.6f}")
    display["pnl_pct"] = display["pnl_pct"].map(lambda x: f"{x:+.4f}%")
    display.columns = ["Entry", "Exit", "Side", "Entry Price", "Exit Price", "PnL (%)", "Hold (min)"]

    def highlight_row(row):
        is_long = "Long" in str(row["Side"])
        color = "rgba(78, 205, 196, 0.15)" if is_long else "rgba(255, 107, 107, 0.15)"
        return [f"background-color: {color}"] * len(row)

    st.dataframe(display.style.apply(highlight_row, axis=1), use_container_width=True, hide_index=True)


@st.cache_data
def backtest_basis_arb(df_hash, window=120, entry_z=2.0, exit_z=0.5, stop_z=4.0, max_hold=360, tx_cost_bps=0.0, leverage=5.0):
    df = load_all()
    basis = np.log(df["btc_usd"].values) - np.log(df["btc_usdt"].values)
    rolling_mean = pd.Series(basis).rolling(window, min_periods=window // 2).mean().values
    rolling_std = pd.Series(basis).rolling(window, min_periods=window // 2).std().values

    trades = []
    position = None
    ts = df["timestamp"].values
    prices = df["btc_usd"].values

    for i in range(window, len(df)):
        if rolling_std[i] == 0 or np.isnan(rolling_std[i]):
            continue
        z = (basis[i] - rolling_mean[i]) / rolling_std[i]

        if position is None:
            if z > entry_z:
                position = {"side": "Short Basis", "entry_idx": i, "entry_price": prices[i]}
            elif z < -entry_z:
                position = {"side": "Long Basis", "entry_idx": i, "entry_price": prices[i]}
        else:
            exit_signal = abs(z) < exit_z or abs(z) > stop_z
            hold_time = i - position["entry_idx"]
            if exit_signal or hold_time > max_hold:
                ep = prices[i]
                raw_pnl = -((ep / position["entry_price"]) - 1) * 100 if position["side"] == "Short Basis" else ((ep / position["entry_price"]) - 1) * 100
                net_pnl = (raw_pnl * leverage) - (tx_cost_bps / 100.0 * 2)
                trades.append({
                    "entry_time": pd.Timestamp(ts[position["entry_idx"]]),
                    "exit_time": pd.Timestamp(ts[i]),
                    "side": position["side"],
                    "entry_price": position["entry_price"],
                    "exit_price": ep,
                    "pnl_pct": net_pnl,
                    "hold_min": hold_time,
                })
                position = None

    return pd.DataFrame(trades)


def compute_risk_metrics(trades_df):
    if trades_df.empty:
        return {}
    pnl = trades_df["pnl_pct"].values
    cum = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cum)
    drawdown = cum - running_max
    n = len(pnl)
    avg = np.mean(pnl)
    std = np.std(pnl, ddof=1) if n > 1 else 0
    downside = pnl[pnl < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else np.nan
    sharpe = (avg / std) * np.sqrt(n) if std > 0 else 0.0
    sortino = (avg / downside_std) * np.sqrt(n) if downside_std and downside_std > 0 else 0.0
    max_dd = np.min(drawdown)
    calmar = cum[-1] / abs(max_dd) if max_dd != 0 else np.inf
    var_95 = np.percentile(pnl, 5)
    var_99 = np.percentile(pnl, 1)
    cvar_95 = np.mean(pnl[pnl <= var_95]) if np.any(pnl <= var_95) else var_95
    cvar_99 = np.mean(pnl[pnl <= var_99]) if np.any(pnl <= var_99) else var_99
    return {
        "Total PnL (%)": cum[-1],
        "Avg PnL/Trade (%)": avg,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Max DD (%)": max_dd,
        "Win Rate (%)": (pnl > 0).mean() * 100,
        "VaR 95% (%)": var_95,
        "CVaR 95% (%)": cvar_95,
        "VaR 99% (%)": var_99,
        "CVaR 99% (%)": cvar_99,
        "# Trades": n,
        "Avg Hold (min)": trades_df["hold_min"].mean(),
    }


def _risk_metrics_display(metrics):
    if not metrics:
        st.warning("No trades generated.")
        return
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total PnL", f"{metrics['Total PnL (%)']:.2f}%")
        st.metric("Win Rate", f"{metrics['Win Rate (%)']:.1f}%")
    with c2:
        st.metric("Sharpe Ratio", f"{metrics['Sharpe']:.2f}")
        st.metric("Sortino Ratio", f"{metrics['Sortino']:.2f}")
    with c3:
        st.metric("Max Drawdown", f"{metrics['Max DD (%)']:.2f}%")
        st.metric("Calmar Ratio", f"{metrics['Calmar']:.2f}")
    with c4:
        st.metric("VaR 95%", f"{metrics['VaR 95% (%)']:.3f}%")
        st.metric("CVaR 95%", f"{metrics['CVaR 95% (%)']:.3f}%")


def _equity_curve_and_drawdown(trades_df):
    if trades_df.empty:
        return
    cum_pnl = trades_df["pnl_pct"].cumsum()
    running_max = cum_pnl.cummax()
    dd = cum_pnl - running_max

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=["Equity Curve (Cumulative PnL %)", "Underwater Plot (Drawdown %)"])
    fig.add_trace(go.Scatter(
        x=trades_df["exit_time"], y=cum_pnl,
        line=dict(color="#4ECDC4", width=1.5), fill="tozeroy",
        fillcolor="rgba(78,205,196,0.1)", name="Cumulative PnL"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=trades_df["exit_time"], y=dd,
        line=dict(color="#FF6B6B", width=1.2), fill="tozeroy",
        fillcolor="rgba(255,107,107,0.15)", name="Drawdown"
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)", row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)", row=2, col=1)
    _chart(fig, "", "", height=500)
    fig.update_yaxes(title_text="Cumulative PnL (%)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)


def _pnl_distribution(trades_df):
    if trades_df.empty:
        return
    pnl = trades_df["pnl_pct"]
    var95 = np.percentile(pnl, 5)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=pnl, nbinsx=30, marker_color="#4ECDC4", opacity=0.8, name="PnL Distribution"))
    fig.add_shape(type="line", x0=var95, x1=var95, y0=0, y1=1, yref="paper",
                  line=dict(color="#FF6B6B", width=2, dash="dash"))
    fig.add_annotation(x=var95, y=0.95, yref="paper", text=f"VaR 95%: {var95:.2f}%",
                       showarrow=True, arrowhead=2, font=dict(color="white", size=11))
    fig.add_shape(type="line", x0=0, x1=0, y0=0, y1=1, yref="paper",
                  line=dict(color="white", width=1))
    _chart(fig, "Trade PnL Distribution", "Frequency", height=350)
    st.plotly_chart(fig, use_container_width=True)


def _robustness_zscore(df_hash):
    df = load_all()
    z_thresholds = np.arange(1.0, 3.75, 0.25)
    results = []
    for z in z_thresholds:
        trades = backtest_basis_arb(df_hash, window=120, entry_z=z, tx_cost_bps=20)
        m = compute_risk_metrics(trades)
        m["Entry Z"] = z
        results.append(m)
    z_df = pd.DataFrame(results)

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["Total PnL vs Z-Threshold", "Sharpe Ratio vs Z-Threshold",
                                        "Max Drawdown vs Z-Threshold", "# Trades vs Z-Threshold"])
    fig.add_trace(go.Scatter(x=z_df["Entry Z"], y=z_df["Total PnL (%)"], mode="lines+markers",
                             line=dict(color="#4ECDC4", width=2), marker=dict(size=7), name="Total PnL"), row=1, col=1)
    fig.add_trace(go.Scatter(x=z_df["Entry Z"], y=z_df["Sharpe"], mode="lines+markers",
                             line=dict(color="#2ca02c", width=2), marker=dict(size=7), name="Sharpe"), row=1, col=2)
    fig.add_trace(go.Scatter(x=z_df["Entry Z"], y=z_df["Max DD (%)"], mode="lines+markers",
                             line=dict(color="#FF6B6B", width=2), marker=dict(size=7), name="Max DD"), row=2, col=1)
    fig.add_trace(go.Scatter(x=z_df["Entry Z"], y=z_df["# Trades"], mode="lines+markers",
                             line=dict(color="#9467bd", width=2), marker=dict(size=7), name="Trades"), row=2, col=2)
    _chart(fig, "Z-Score Threshold Sensitivity", "", height=550)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(z_df[["Entry Z", "Total PnL (%)", "Sharpe", "Max DD (%)", "Win Rate (%)", "# Trades"]].round(3),
                 use_container_width=True, hide_index=True)


def _robustness_window(df_hash):
    windows = [30, 60, 90, 120, 180, 240, 360, 480]
    results = []
    for w in windows:
        trades = backtest_basis_arb(df_hash, window=w, tx_cost_bps=20)
        m = compute_risk_metrics(trades)
        m["Window"] = w
        results.append(m)
    w_df = pd.DataFrame(results)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Sharpe Ratio vs Window Length", "Total PnL vs Window Length"])
    fig.add_trace(go.Bar(x=w_df["Window"].astype(str), y=w_df["Sharpe"],
                         marker_color="#2ca02c", name="Sharpe"), row=1, col=1)
    fig.add_trace(go.Bar(x=w_df["Window"].astype(str), y=w_df["Total PnL (%)"],
                         marker_color="#4ECDC4", name="Total PnL"), row=1, col=2)
    _chart(fig, "Rolling Window Sensitivity", "", height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(w_df[["Window", "Total PnL (%)", "Sharpe", "Max DD (%)", "Win Rate (%)", "# Trades"]].round(3),
                 use_container_width=True, hide_index=True)


def _robustness_txcost(df_hash):
    costs = np.arange(0, 65, 5)
    results = []
    for c in costs:
        trades = backtest_basis_arb(df_hash, tx_cost_bps=c)
        m = compute_risk_metrics(trades)
        m["Tx Cost (bps)"] = c
        results.append(m)
    c_df = pd.DataFrame(results)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c_df["Tx Cost (bps)"], y=c_df["Total PnL (%)"],
                             mode="lines+markers", line=dict(color="#4ECDC4", width=2),
                             marker=dict(size=8), name="Total PnL"))
    fig.add_trace(go.Scatter(x=c_df["Tx Cost (bps)"], y=c_df["Sharpe"],
                             mode="lines+markers", line=dict(color="#2ca02c", width=2),
                             marker=dict(size=8), name="Sharpe Ratio", yaxis="y2"))
    fig.add_hline(y=0, line_dash="dash", line_color="white", line_width=1)
    _chart(fig, "Transaction Cost Sensitivity: Break-Even Analysis", "Total PnL (%)", height=420)
    fig.update_layout(yaxis2=dict(title="Sharpe Ratio", overlaying="y", side="right", showgrid=False))
    st.plotly_chart(fig, use_container_width=True)

    breakeven = c_df[c_df["Total PnL (%)"] <= 0]
    if not breakeven.empty:
        st.info(f"💡 **Break-even transaction cost:** ~{breakeven.iloc[0]['Tx Cost (bps)']:.0f} bps")
    else:
        st.success("Strategy remains profitable across all tested cost levels (0–60 bps)")

    st.dataframe(c_df[["Tx Cost (bps)", "Total PnL (%)", "Sharpe", "Win Rate (%)", "# Trades"]].round(3),
                 use_container_width=True, hide_index=True)


def _structural_break(df):
    svb_date = pd.Timestamp("2023-03-10")
    if df["timestamp"].dt.tz is not None:
        svb_date = svb_date.tz_localize(df["timestamp"].dt.tz)

    basis_bps = (np.log(df["btc_usd"]) - np.log(df["btc_usdt"])) * 10000
    pre = basis_bps[df["timestamp"] < svb_date].dropna()
    post = basis_bps[df["timestamp"] >= svb_date].dropna()
    full = basis_bps.dropna()

    def ssr(series):
        x = np.arange(len(series)).reshape(-1, 1)
        y = series.values
        A = np.column_stack([x, np.ones(len(x))])
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        predicted = A @ coeffs
        return np.sum((y - predicted) ** 2)

    ssr_full = ssr(full)
    ssr_pre = ssr(pre)
    ssr_post = ssr(post)
    k = 2
    n = len(full)
    chow_stat = ((ssr_full - ssr_pre - ssr_post) / k) / ((ssr_pre + ssr_post) / (n - 2 * k))
    p_value = 1 - stats.f.cdf(chow_stat, k, n - 2 * k)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Chow F-Statistic", f"{chow_stat:.2f}")
    with c2:
        st.metric("p-value", f"{p_value:.2e}")
    with c3:
        st.metric("Significant at 1%", "YES ✅" if p_value < 0.01 else "NO ❌")

    pre_ts = df[df["timestamp"] < svb_date]["timestamp"]
    post_ts = df[df["timestamp"] >= svb_date]["timestamp"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pre_ts, y=pre.values, line=dict(color="#4ECDC4", width=0.8), name="Pre-Crisis"))
    fig.add_trace(go.Scatter(x=post_ts, y=post.values, line=dict(color="#FF6B6B", width=0.8), name="Post-Crisis"))
    fig.add_shape(type="line", x0=str(svb_date), x1=str(svb_date), y0=0, y1=1, yref="paper",
                  line=dict(color="white", width=2, dash="dash"))
    fig.add_annotation(x=str(svb_date), y=1.05, yref="paper",
                       text=f"SVB Break (F={chow_stat:.1f})", showarrow=False,
                       font=dict(size=12, color="white"))
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
    _chart(fig, "Structural Break Detection: Chow Test on Basis Series", "Basis (bps)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    **Interpretation:** The basis series exhibits a statistically significant structural break at the SVB crisis onset.

    |  | Pre-Crisis | Post-Crisis |
    |---|---|---|
    | **Mean** | {pre.mean():.2f} bps | {post.mean():.2f} bps |
    | **Std Dev** | {pre.std():.2f} bps | {post.std():.2f} bps |
    """)


def render():
    df = load_all()
    df_5m = downsample(df, "5min")
    df_hash = hash(len(df))

    st.header("Quantitative Strategies")

    st.markdown("""
    **Index:**
    1. [Cross-Currency Basis Arbitrage](#cross-currency-basis-arbitrage)
    2. [Robustness & Sensitivity Analysis](#robustness-sensitivity-analysis)
    """)

    st.divider()

    st.header("Cross-Currency Basis Arbitrage")
    st.markdown(
        """
        Exploits the decoupling between BTC/USD and BTC/USDT during stress. When the basis 
        deviates beyond a Z-score threshold, we enter a convergence trade expecting mean-reversion.
        """
    )
    st.latex(r"Z_t = \frac{Basis_t - \mu_{W}}{\sigma_{W}}, \quad Basis_t = \ln(P_{BTC/USD}) - \ln(P_{BTC/USDT})")

    with st.expander("⚙️ Strategy Parameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            window = st.slider("Rolling Window (min)", 30, 360, 120, 30)
            leverage = st.slider("Leverage (×)", 1.0, 10.0, 5.0, 0.5)
        with c2:
            entry_z = st.slider("Entry Z-Score", 1.0, 4.0, 2.0, 0.25)
            exit_z = st.slider("Target Exit Z-Score", 0.0, 2.0, 0.5, 0.1)
        with c3:
            stop_z = st.slider("Stop-Loss Z-Score", 2.0, 6.0, 4.0, 0.5)
            tx_cost = st.slider("Tx Cost (bps round-trip)", 0.0, 60.0, 20.0, 5.0)

    trades = backtest_basis_arb(
        df_hash, window=window, entry_z=entry_z, exit_z=exit_z,
        stop_z=stop_z, max_hold=360, tx_cost_bps=tx_cost, leverage=leverage
    )
    metrics = compute_risk_metrics(trades)

    fig = _price_chart_with_signals(df_5m, trades, "btc_usd", "BTC/USD — Basis Arb Signals")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Risk Decomposition")
    _risk_metrics_display(metrics)

    col1, col2 = st.columns([2, 1])
    with col1:
        _equity_curve_and_drawdown(trades)
    with col2:
        _pnl_distribution(trades)

    st.markdown("""
    **Funding & Execution Assumptions:**
    - Maker/Taker: ~5–10 bps/leg · Slippage: ~2–5 bps/leg · Gas: ~$2–5/transfer
    - Margin rate: ~8% annualized · Weekend settlement risk on fiat leg
    """)

    st.divider()
    st.subheader("Trade Ledger")
    _order_book(trades)

    st.divider()

    st.header("Robustness & Sensitivity Analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Z-Score Sensitivity",
        "📏 Window Length Sensitivity",
        "💰 Transaction Cost Break-Even",
        "🔬 Structural Break (Chow Test)"
    ])

    with tab1:
        st.markdown("Sweep entry Z-score from 1.0 → 3.5 to ensure the baseline Z=2.0 is not an artifact.")
        with st.spinner("Running Z-score sensitivity sweep..."):
            _robustness_zscore(df_hash)

    with tab2:
        st.markdown("Test rolling windows from 30 to 480 min to verify 120-min baseline is robust.")
        with st.spinner("Running window length sweep..."):
            _robustness_window(df_hash)

    with tab3:
        st.markdown("Sweep round-trip costs from 0 → 60 bps to find the break-even transaction cost.")
        with st.spinner("Running transaction cost sweep..."):
            _robustness_txcost(df_hash)

    with tab4:
        st.markdown("Chow test for a formal structural break in the basis at the SVB crisis onset (March 10, 2023).")
        with st.spinner("Running Chow test..."):
            _structural_break(df)
