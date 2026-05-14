"""
eda.py — NSE-Compatible Stock Exploratory Data Analysis Pipeline
===============================================================
A modular, function-based toolkit for:
  • Data fetching & normalization (NSE / BSE / US via yfinance)
  • Technical indicators
  • Statistical analysis (statsmodels-based)
  • Time-series decomposition, ADF, ARIMA, SARIMA
  • Residual diagnostics
  • Visualization helpers

Usage:
    from eda import *
    df = fetch_data("RELIANCE.NS", period="2y")
    df = add_technicals(df)
    stats = compute_summary_stats(df)
    plot_candlestick(df)
    run_stationarity_tests(df['Close'])
    model, forecast = fit_arima(df['Close'])
    model_s, fc_s = fit_sarima(df['Close'])
    diagnose_residuals(model.resid)

Author: Generated pipeline
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf

# Stats & Time Series
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import (
    acorr_ljungbox, het_arch, jarque_bera
)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# ---------------------------------------------------------------------------
# SECTION 1: DATA FETCHING & NORMALIZATION
# ---------------------------------------------------------------------------

def fetch_data(ticker: str, period: str = "1y", interval: str = "1d",
               start: str = None, end: str = None,
               auto_adjust: bool = True, progress: bool = False) -> pd.DataFrame:
    """
    Fetch OHLCV data from yfinance. Works for NSE (.NS), BSE (.BO), US, crypto, indices.

    Parameters
    ----------
    ticker : str
        Symbol. NSE: "RELIANCE.NS", BSE: "RELIANCE.BO", US: "AAPL", Index: "^NSEI"
    period : str
        yfinance period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    interval : str
        1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    start, end : str (YYYY-MM-DD)
        Absolute date range. Overrides `period` if both provided.
    auto_adjust : bool
        If True, returns adjusted close in the 'Close' column (splits & dividends).

    Returns
    -------
    pd.DataFrame with columns: Date(index), Open, High, Low, Close, Volume, [Adj Close]
    """
    df = yf.download(
        ticker,
        period=period if (start is None or end is None) else None,
        start=start, end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=progress
    )

    # Flatten MultiIndex columns if single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure standard column names (case-insensitive safety)
    df.columns = [c.title() for c in df.columns]

    # Drop rows with all NaNs
    df = df.dropna(how="all")

    # Ensure Date is index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Sort chronologically
    df = df.sort_index()

    return df


def normalize_ticker(symbol: str, exchange: str = "NSE") -> str:
    """
    Append exchange suffix if missing.

    Examples
    --------
    normalize_ticker("RELIANCE", "NSE") -> "RELIANCE.NS"
    normalize_ticker("RELIANCE", "BSE") -> "RELIANCE.BO"
    normalize_ticker("AAPL", "US")        -> "AAPL"
    normalize_ticker("^NSEI", "NSE")    -> "^NSEI"
    """
    symbol = symbol.strip().upper()
    if exchange.upper() == "NSE" and not symbol.endswith(".NS") and not symbol.startswith("^"):
        return f"{symbol}.NS"
    if exchange.upper() == "BSE" and not symbol.endswith(".BO") and not symbol.startswith("^"):
        return f"{symbol}.BO"
    return symbol


def resample_data(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """
    Resample OHLCV to a lower frequency.

    Parameters
    ----------
    freq : str
        'W' (weekly), 'M' (monthly), 'Q' (quarterly), 'Y' (yearly)
    """
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }
    # Only aggregate columns that exist
    agg = {k: v for k, v in agg.items() if k in df.columns}
    freq_map = {"M": "ME", "Q": "QE", "Y": "YE"}
    freq = freq_map.get(freq, freq)
    return df.resample(freq).agg(agg).dropna()


# ---------------------------------------------------------------------------
# SECTION 2: TECHNICAL INDICATORS
# ---------------------------------------------------------------------------

def add_returns(df: pd.DataFrame, col: str = "Close") -> pd.DataFrame:
    """Add daily & log returns."""
    df = df.copy()
    df["Daily_Return"] = df[col].pct_change()
    df["Log_Return"] = np.log(df[col] / df[col].shift(1))
    df["Cumulative_Return"] = (1 + df["Daily_Return"]).cumprod() - 1
    return df


def add_moving_averages(df: pd.DataFrame, col: str = "Close",
                        windows: list = [10, 20, 50, 200]) -> pd.DataFrame:
    """Add SMA and EMA for given windows."""
    df = df.copy()
    for w in windows:
        df[f"SMA_{w}"] = df[col].rolling(window=w).mean()
        df[f"EMA_{w}"] = df[col].ewm(span=w, adjust=False).mean()
    return df


def add_bollinger_bands(df: pd.DataFrame, col: str = "Close",
                        window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Add Bollinger Bands."""
    df = df.copy()
    df["BB_Middle"] = df[col].rolling(window=window).mean()
    std = df[col].rolling(window=window).std()
    df["BB_Upper"] = df["BB_Middle"] + num_std * std
    df["BB_Lower"] = df["BB_Middle"] - num_std * std
    df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]
    df["BB_Position"] = (df[col] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])
    return df


def add_rsi(df: pd.DataFrame, col: str = "Close", window: int = 14) -> pd.DataFrame:
    """Add Relative Strength Index."""
    df = df.copy()
    delta = df[col].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame, col: str = "Close",
             fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Add MACD, Signal line, and Histogram."""
    df = df.copy()
    ema_fast = df[col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[col].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Add Average True Range."""
    df = df.copy()
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window=window).mean()
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Add On-Balance Volume."""
    df = df.copy()
    sign = np.sign(df["Close"].diff())
    df["OBV"] = (sign * df["Volume"]).cumsum()
    return df


def add_volatility(df: pd.DataFrame, col: str = "Close",
                   windows: list = [5, 10, 20, 60]) -> pd.DataFrame:
    """Add rolling realized volatility (annualized)."""
    df = df.copy()
    for w in windows:
        df[f"Vol_{w}d"] = df[col].pct_change().rolling(window=w).std() * np.sqrt(252)
    return df


def add_all_technicals(df: pd.DataFrame, col: str = "Close") -> pd.DataFrame:
    """Run all technical indicator functions."""
    df = add_returns(df, col)
    df = add_moving_averages(df, col)
    df = add_bollinger_bands(df, col)
    df = add_rsi(df, col)
    df = add_macd(df, col)
    df = add_atr(df)
    df = add_obv(df)
    df = add_volatility(df, col)
    return df


# ---------------------------------------------------------------------------
# SECTION 3: STATISTICAL ANALYSIS (statsmodels)
# ---------------------------------------------------------------------------

def compute_summary_stats(series: pd.Series) -> dict:
    """
    Comprehensive descriptive statistics.
    """
    s = series.dropna()
    return {
        "count": len(s),
        "mean": s.mean(),
        "median": s.median(),
        "std": s.std(),
        "variance": s.var(),
        "min": s.min(),
        "max": s.max(),
        "range": s.max() - s.min(),
        "skewness": s.skew(),
        "kurtosis": s.kurtosis(),
        "jarque_bera_stat": jarque_bera(s)[0],
        "jarque_bera_pvalue": jarque_bera(s)[1],
        "shapiro_wilk_stat": stats.shapiro(s.iloc[:min(5000, len(s))])[0],
        "shapiro_wilk_pvalue": stats.shapiro(s.iloc[:min(5000, len(s))])[1],
        "percentile_5": s.quantile(0.05),
        "percentile_25": s.quantile(0.25),
        "percentile_75": s.quantile(0.75),
        "percentile_95": s.quantile(0.95),
        "cv": s.std() / s.mean() if s.mean() != 0 else np.nan,
    }


def run_stationarity_tests(series: pd.Series, significance: float = 0.05) -> dict:
    """
    ADF and KPSS stationarity tests.

    Returns dict with test stats, p-values, and conclusion.
    """
    s = series.dropna()

    # ADF
    adf_result = adfuller(s, autolag="AIC")
    adf_stat, adf_pvalue = adf_result[0], adf_result[1]
    adf_conclusion = "Stationary" if adf_pvalue < significance else "Non-Stationary"

    # KPSS
    kpss_result = kpss(s, regression="c", nlags="auto")
    kpss_stat, kpss_pvalue = kpss_result[0], kpss_result[1]
    kpss_conclusion = "Non-Stationary" if kpss_pvalue < significance else "Stationary"

    return {
        "adf_statistic": adf_stat,
        "adf_pvalue": adf_pvalue,
        "adf_conclusion": adf_conclusion,
        "adf_critical_values": adf_result[4],
        "kpss_statistic": kpss_stat,
        "kpss_pvalue": kpss_pvalue,
        "kpss_conclusion": kpss_conclusion,
        "kpss_critical_values": kpss_result[3],
        "both_agree_stationary": (adf_conclusion == "Stationary" and kpss_conclusion == "Stationary"),
        "both_agree_nonstationary": (adf_conclusion == "Non-Stationary" and kpss_conclusion == "Non-Stationary"),
    }


def decompose_series(series: pd.Series, model: str = "additive",
                     period: int = None, plot: bool = False):
    """
    Seasonal decomposition (trend, seasonal, residual).

    Parameters
    ----------
    model : "additive" or "multiplicative"
    period : int
        Seasonal period. If None, inferred from frequency.
    plot : bool
        If True, returns matplotlib figure (for notebook display).
    """
    s = series.dropna()

    if period is None:
        # Infer from index frequency
        freq = pd.infer_freq(s.index)
        if freq in ["D", "B"]:
            period = 252  # Trading days per year
        elif freq in ["W", "W-FRI"]:
            period = 52
        elif freq in ["M", "MS"]:
            period = 12
        else:
            period = 252  # default

    result = seasonal_decompose(s, model=model, period=period, extrapolate_trend="freq")

    if plot:
        fig = result.plot()
        fig.set_size_inches(12, 8)
        return result, fig

    return result


def fit_arima(series: pd.Series, order: tuple = (1, 1, 1),
              seasonal_order: tuple = None,
              forecast_steps: int = 30,
              conf_alpha: float = 0.05) -> tuple:
    """
    Fit ARIMA or SARIMA model.

    Parameters
    ----------
    order : tuple (p, d, q)
    seasonal_order : tuple (P, D, Q, s) or None for plain ARIMA
    forecast_steps : int
        Number of steps to forecast ahead.
    conf_alpha : float
        Confidence level for prediction intervals.

    Returns
    -------
    (model_fit, forecast_df)
    """
    s = series.dropna()

    if seasonal_order is not None:
        model = SARIMAX(s, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
    else:
        model = ARIMA(s, order=order)

    model_fit = model.fit()

    # Forecast
    forecast = model_fit.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=conf_alpha)

    # Build forecast DataFrame
    last_date = s.index[-1]
    freq = pd.infer_freq(s.index) or "B"
    future_dates = pd.date_range(start=last_date, periods=forecast_steps + 1, freq=freq)[1:]

    forecast_df = pd.DataFrame({
        "forecast": forecast_mean.values,
        "lower_ci": conf_int.iloc[:, 0].values,
        "upper_ci": conf_int.iloc[:, 1].values,
    }, index=future_dates)

    return model_fit, forecast_df


def fit_sarima(series: pd.Series,
               order: tuple = (1, 1, 1),
               seasonal_order: tuple = (1, 1, 1, 12),
               forecast_steps: int = 30,
               conf_alpha: float = 0.05) -> tuple:
    """
    Convenience wrapper for SARIMA with explicit seasonal component.
    For daily stock data, use seasonal_order=(1,1,1,252) or (1,1,1,5) for weekly.
    """
    return fit_arima(series, order=order, seasonal_order=seasonal_order,
                     forecast_steps=forecast_steps, conf_alpha=conf_alpha)


def diagnose_residuals(residuals: pd.Series, lags: int = 10) -> dict:
    """
    Residual diagnostics: Ljung-Box, ARCH-LM, Jarque-Bera, normality.

    Returns dict of test results.
    """
    r = residuals.dropna()

    # Ljung-Box (autocorrelation in residuals)
    lb = acorr_ljungbox(r, lags=lags, return_df=True)
    lb_conclusion = "No autocorrelation" if lb["lb_pvalue"].iloc[-1] > 0.05 else "Autocorrelation detected"

    # ARCH-LM (heteroskedasticity / volatility clustering)
    arch_lm = het_arch(r, maxlag=lags)
    arch_stat, arch_pvalue = arch_lm[0], arch_lm[1]
    arch_conclusion = "No ARCH effects" if arch_pvalue > 0.05 else "ARCH effects detected"

    # Jarque-Bera (normality)
    jb_stat, jb_pvalue = jarque_bera(r)
    jb_conclusion = "Residuals normal" if jb_pvalue > 0.05 else "Residuals non-normal"

    return {
        "ljung_box_stat": lb["lb_stat"].iloc[-1],
        "ljung_box_pvalue": lb["lb_pvalue"].iloc[-1],
        "ljung_box_conclusion": lb_conclusion,
        "arch_lm_stat": arch_stat,
        "arch_lm_pvalue": arch_pvalue,
        "arch_lm_conclusion": arch_conclusion,
        "jarque_bera_stat": jb_stat,
        "jarque_bera_pvalue": jb_pvalue,
        "jarque_bera_conclusion": jb_conclusion,
        "residual_mean": r.mean(),
        "residual_std": r.std(),
        "residual_skew": r.skew(),
        "residual_kurtosis": r.kurtosis(),
    }


def compute_vif(df: pd.DataFrame, features: list = None) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor for multicollinearity check.
    Useful when building regression models with technical indicators.
    """
    df_num = df.select_dtypes(include=[np.number]).dropna()
    if features:
        df_num = df_num[features]

    vif_data = pd.DataFrame()
    vif_data["feature"] = df_num.columns
    vif_data["VIF"] = [variance_inflation_factor(df_num.values, i) for i in range(df_num.shape[1])]
    return vif_data


# ---------------------------------------------------------------------------
# SECTION 4: VISUALIZATION HELPERS (Plotly)
# ---------------------------------------------------------------------------

def plot_candlestick(df: pd.DataFrame, ticker: str = "Stock",
                     show_ma: bool = True, show_bb: bool = True,
                     height: int = 600) -> go.Figure:
    """
    Interactive candlestick chart with optional MA and Bollinger Bands.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.7, 0.3],
                        subplot_titles=(f"{ticker} Price", "Volume"))

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name=ticker,
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350"
    ), row=1, col=1)

    if show_ma and "SMA_20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], mode="lines",
            name="SMA20", line=dict(color="#ff9800", width=1.5)), row=1, col=1)
    if show_ma and "SMA_50" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], mode="lines",
            name="SMA50", line=dict(color="#2196f3", width=1.5)), row=1, col=1)

    if show_bb and "BB_Upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], mode="lines",
            name="BB Upper", line=dict(color="gray", width=1, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], mode="lines",
            name="BB Lower", line=dict(color="gray", width=1, dash="dash"),
            fill="tonexty", fillcolor="rgba(128,128,128,0.1)"), row=1, col=1)

    colors = ["#26a69a" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#ef5350"
              for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
        marker_color=colors, opacity=0.6), row=2, col=1)

    fig.update_layout(
        title=dict(text=f"<b>{ticker} Candlestick Chart</b>", x=0.5),
        xaxis_rangeslider_visible=False,
        height=height,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_returns_distribution(df: pd.DataFrame, col: str = "Daily_Return",
                               ticker: str = "Stock", height: int = 500) -> go.Figure:
    """
    Histogram of returns with normal overlay, QQ plot, and box plot.
    """
    returns = df[col].dropna()

    fig = make_subplots(rows=1, cols=3,
        subplot_titles=("Histogram + Normal", "Box Plot", "Q-Q Plot"),
        specs=[[{}, {"type": "box"}, {}]])

    # Histogram
    fig.add_trace(go.Histogram(x=returns, nbinsx=50, name="Returns",
        marker_color="#2196f3", opacity=0.7, histnorm="probability density"), row=1, col=1)

    x_range = np.linspace(returns.min(), returns.max(), 200)
    normal_pdf = stats.norm.pdf(x_range, returns.mean(), returns.std())
    fig.add_trace(go.Scatter(x=x_range, y=normal_pdf, mode="lines",
        name="Normal", line=dict(color="#e91e63", width=2)), row=1, col=1)

    fig.add_vline(x=returns.mean(), line_dash="dash", line_color="#4caf50",
        annotation_text=f"μ={returns.mean():.3f}", row=1, col=1)

    # Box
    fig.add_trace(go.Box(y=returns, name="Returns", boxmean="sd",
        marker_color="#9c27b0"), row=1, col=2)

    # Q-Q (theoretical vs sample quantiles)
    theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(returns)))
    sample_sorted = np.sort(returns)
    fig.add_trace(go.Scatter(x=theoretical, y=sample_sorted, mode="markers",
        name="Q-Q", marker=dict(color="#673ab7", size=4, opacity=0.6)), row=1, col=3)

    # Reference line
    q_min, q_max = theoretical.min(), theoretical.max()
    fig.add_trace(go.Scatter(x=[q_min, q_max], y=[q_min*returns.std()+returns.mean(), q_max*returns.std()+returns.mean()],
        mode="lines", name="Reference", line=dict(color="red", dash="dash")), row=1, col=3)

    fig.update_layout(
        title=dict(text=f"<b>{ticker} Returns Distribution Analysis</b>", x=0.5),
        height=height, template="plotly_white", showlegend=False
    )
    return fig


def plot_decomposition(result, ticker: str = "Stock", height: int = 800) -> go.Figure:
    """
    Plot seasonal decomposition results interactively.
    """
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))

    fig.add_trace(go.Scatter(x=result.observed.index, y=result.observed,
        mode="lines", name="Observed", line=dict(color="#2196f3")), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.trend.index, y=result.trend,
        mode="lines", name="Trend", line=dict(color="#ff9800")), row=2, col=1)
    fig.add_trace(go.Scatter(x=result.seasonal.index, y=result.seasonal,
        mode="lines", name="Seasonal", line=dict(color="#4caf50")), row=3, col=1)
    fig.add_trace(go.Scatter(x=result.resid.index, y=result.resid,
        mode="lines", name="Residual", line=dict(color="#9c27b0")), row=4, col=1)

    fig.update_layout(
        title=dict(text=f"<b>{ticker} Time-Series Decomposition</b>", x=0.5),
        height=height, template="plotly_white", hovermode="x unified",
        showlegend=False
    )
    return fig


def plot_forecast(series: pd.Series, forecast_df: pd.DataFrame,
                  ticker: str = "Stock", height: int = 500) -> go.Figure:
    """
    Plot historical series + forecast with confidence intervals.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=series.index, y=series, mode="lines",
        name="Historical", line=dict(color="#2196f3", width=1.5)))

    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["forecast"], mode="lines",
        name="Forecast", line=dict(color="#ff9800", width=2)))

    fig.add_trace(go.Scatter(
        x=list(forecast_df.index) + list(forecast_df.index[::-1]),
        y=list(forecast_df["upper_ci"]) + list(forecast_df["lower_ci"][::-1]),
        fill="toself", fillcolor="rgba(255,152,0,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Confidence Interval", hoverinfo="skip"
    ))

    fig.update_layout(
        title=dict(text=f"<b>{ticker} ARIMA/SARIMA Forecast</b>", x=0.5),
        height=height, template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_acf_pacf(series: pd.Series, lags: int = 40, height: int = 500):
    """
    Generate ACF and PACF plots using statsmodels (returns matplotlib figures).
    For interactive use in notebooks.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series.dropna(), lags=lags, ax=axes[0], title="ACF")
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], title="PACF", method="ywm")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# SECTION 5: PIPELINE ORCHESTRATOR
# ---------------------------------------------------------------------------

def full_pipeline(ticker: str, exchange: str = "NSE",
                  period: str = "2y", interval: str = "1d",
                  arima_order: tuple = (1, 1, 1),
                  sarima_seasonal: tuple = (1, 1, 1, 252),
                  forecast_steps: int = 30,
                  verbose: bool = True) -> dict:
    """
    One-call full EDA + modeling pipeline.

    Returns dict with all results for programmatic access.
    """
    # 1. Fetch
    sym = normalize_ticker(ticker, exchange)
    df = fetch_data(sym, period=period, interval=interval)

    # 2. Technicals
    df = add_all_technicals(df)

    # 3. Summary stats
    price_stats = compute_summary_stats(df["Close"])
    return_stats = compute_summary_stats(df["Daily_Return"])

    # 4. Stationarity
    stationarity = run_stationarity_tests(df["Close"])

    # 5. Decomposition
    decomp = decompose_series(df["Close"], model="additive", period=252)

    # 6. ARIMA
    arima_fit, arima_fc = fit_arima(df["Close"], order=arima_order,
                                     forecast_steps=forecast_steps)
    arima_diag = diagnose_residuals(arima_fit.resid)

    # 7. SARIMA
    sarima_fit, sarima_fc = fit_sarima(df["Close"], order=arima_order,
                                        seasonal_order=sarima_seasonal,
                                        forecast_steps=forecast_steps)
    sarima_diag = diagnose_residuals(sarima_fit.resid)

    # 8. VIF (multicollinearity of indicators)
    tech_cols = [c for c in df.columns if any(x in c for x in ["SMA", "EMA", "RSI", "MACD", "ATR", "BB"])]
    vif_df = compute_vif(df, tech_cols) if tech_cols else None

    if verbose:
        print(f"\n{'='*60}")
        print(f"PIPELINE COMPLETE: {sym}")
        print(f"{'='*60}")
        print(f"Data points: {len(df)} | Date range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"Current Close: {df['Close'].iloc[-1]:.2f}")
        print(f"ADF: {stationarity['adf_conclusion']} (p={stationarity['adf_pvalue']:.4f})")
        print(f"KPSS: {stationarity['kpss_conclusion']} (p={stationarity['kpss_pvalue']:.4f})")
        print(f"ARIMA AIC: {arima_fit.aic:.2f} | SARIMA AIC: {sarima_fit.aic:.2f}")
        print(f"ARIMA resid Ljung-Box: {arima_diag['ljung_box_conclusion']}")
        print(f"SARIMA resid Ljung-Box: {sarima_diag['ljung_box_conclusion']}")

    return {
        "ticker": sym,
        "df": df,
        "price_stats": price_stats,
        "return_stats": return_stats,
        "stationarity": stationarity,
        "decomposition": decomp,
        "arima": {"model": arima_fit, "forecast": arima_fc, "diagnostics": arima_diag},
        "sarima": {"model": sarima_fit, "forecast": sarima_fc, "diagnostics": sarima_diag},
        "vif": vif_df,
    }


# ---------------------------------------------------------------------------
# __all__ for clean imports
# ---------------------------------------------------------------------------
__all__ = [
    # Data
    "fetch_data", "normalize_ticker", "resample_data",
    # Technicals
    "add_returns", "add_moving_averages", "add_bollinger_bands",
    "add_rsi", "add_macd", "add_atr", "add_obv", "add_volatility",
    "add_all_technicals",
    # Stats
    "compute_summary_stats", "run_stationarity_tests",
    "decompose_series", "fit_arima", "fit_sarima",
    "diagnose_residuals", "compute_vif",
    # Viz
    "plot_candlestick", "plot_returns_distribution",
    "plot_decomposition", "plot_forecast", "plot_acf_pacf",
    # Pipeline
    "full_pipeline",
]
