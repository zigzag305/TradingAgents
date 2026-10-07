"""Deterministic market-data verification snapshot.

The market analyst is an LLM that can confabulate exact numbers — citing a
Bollinger band or a "historically validated bounce" that the underlying data
doesn't support (#830). This module computes a ground-truth snapshot (latest
OHLCV row on or before the analysis date, common indicators, recent closes)
the analyst is told to treat as the source of truth for any exact numeric
claim. Deterministic, no LLM involved.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd
from stockstats import wrap

from tradingagents.dataflows.stockstats_utils import load_ohlcv

# A fixed, common indicator set so the snapshot is the same shape every run.
DEFAULT_SNAPSHOT_INDICATORS: tuple[str, ...] = (
    "close_10_ema", "close_50_sma", "close_200_sma",
    "rsi", "boll", "boll_ub", "boll_lb",
    "macd", "macds", "macdh", "atr",
)

# Prior-row windows the latest session is compared against for range breaks.
STRUCTURE_RANGE_WINDOWS: tuple[int, ...] = (5, 10)
# 50/200-day gap (in ATRs) needed before their order is read as a trend; in a
# sideways market the averages weave around each other.
TREND_SEPARATION_ATR = 0.5


_WINDOWED = re.compile(r"^close_(\d+)_(?:sma|ema)$")


# Fixed-window indicators the structure section depends on (ATR(14) needs 15 closes).
_FIXED_WINDOWS = {"atr": 15}


def _required_rows(indicator: str) -> int:
    """Rows an indicator needs; stockstats fills shorter histories with a
    partial-window value, which would pass a 30-row average off as a 200-day one."""
    m = _WINDOWED.match(indicator)
    return int(m.group(1)) if m else _FIXED_WINDOWS.get(indicator, 0)


def _today_dates() -> set:
    """Current calendar date in UTC and locally (crypto bars are UTC days)."""
    return {pd.Timestamp.now(tz="UTC").date(), pd.Timestamp.today().date()}


def _num(frame, column: str, row: int = -1):
    if _required_rows(column) + (-row - 1) > len(frame):
        return None
    try:
        value = float(frame[column].iloc[row])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return None if pd.isna(value) else value


def _structure_lines(df: pd.DataFrame, stock_df) -> list[str]:
    """Computed facts about the latest session and where it sits in recent structure.

    The closing guidance tells the analyst not to assert percentage moves or
    support/resistance breaks unless tool output states them -- yet the raw rows
    never state them. On a day that breaks a range, the analyst could not say
    so, and the debate drifted toward a hedged Hold. These lines are derived
    from the same verified rows, so they can be cited.
    """
    if len(df) < 2:
        return []
    latest, prev = df.iloc[-1], df.iloc[-2]
    close, prev_close = float(latest["Close"]), float(prev["Close"])
    high, low = float(latest["High"]), float(latest["Low"])
    day = pd.Timestamp(latest["Date"]).date()
    # Today's move is scaled by the ATR *before* today, so a big session does
    # not inflate the yardstick it is measured with.
    atr = _num(stock_df, "atr", -2)
    has_atr = atr is not None and atr > 0

    def in_atr(delta: float) -> str:
        return f" ({delta / atr:+.2f} x ATR(14) of {atr:.2f})" if has_atr else ""

    lines = ["", "### Session & structure (computed from the rows above)", ""]
    if day in _today_dates():
        lines.append(f"- {day} is the current day, so this row may still be forming: its close, "
                     "range and the range checks that follow are provisional until the session closes.")
    lines.append(f"- Latest session {day}: close {close:.2f} vs prior close {prev_close:.2f}, "
                 f"{(close / prev_close - 1) * 100:+.2f}%{in_atr(close - prev_close)}.")
    if high > low:
        third = (close - low) / (high - low)
        where = "lower" if third < 1 / 3 else "upper" if third > 2 / 3 else "middle"
        lines.append(f"- Session range {low:.2f}-{high:.2f}; the close sits in the {where} third.")

    range_state = None
    for n in STRUCTURE_RANGE_WINDOWS:
        prior = df.iloc[-1 - n:-1]
        if len(prior) < n:
            continue
        p_hi, p_lo = float(prior["High"].max()), float(prior["Low"].min())
        if close < p_lo:
            state = f"closed BELOW the prior {n}-row low {p_lo:.2f} (range breakdown)"
        elif close > p_hi:
            state = f"closed ABOVE the prior {n}-row high {p_hi:.2f} (range breakout)"
        elif low < p_lo:
            state = f"traded below the prior {n}-row low {p_lo:.2f} but closed back inside (failed breakdown)"
        elif high > p_hi:
            state = f"traded above the prior {n}-row high {p_hi:.2f} but closed back inside (failed breakout)"
        else:
            state = f"stayed inside the prior {n}-row range {p_lo:.2f}-{p_hi:.2f}"
        lines.append(f"- Versus the prior {n} rows: {state}.")
        range_state = state

    for column, label in (("close_20_sma", "20-day SMA"), ("close_50_sma", "50-day SMA"),
                          ("close_200_sma", "200-day SMA")):
        value = _num(stock_df, column)
        if value is None:
            continue
        side = "above" if close > value else "below"
        lines.append(f"- {label} {value:.2f}: close is {abs(close / value - 1) * 100:.2f}% {side} it"
                     f"{in_atr(close - value)}.")

    sma50, sma200 = _num(stock_df, "close_50_sma"), _num(stock_df, "close_200_sma")
    sma50_then = _num(stock_df, "close_50_sma", -11)
    if sma50 is None or sma200 is None or sma50_then is None or not has_atr:
        regime = "not enough history for a 50/200-day trend read"
    elif abs(sma50 - sma200) < TREND_SEPARATION_ATR * atr:
        regime = "no trend: the 50-day and 200-day have converged"
    elif close > sma50 > sma200 and sma50 > sma50_then:
        regime = "uptrend (close above a rising 50-day, 50-day above the 200-day)"
    elif close < sma50 < sma200 and sma50 < sma50_then:
        regime = "downtrend (close below a falling 50-day, 50-day below the 200-day)"
    elif close < sma50 and sma50 > sma200:
        regime = "correction inside a longer uptrend (close lost the 50-day)"
    elif close > sma50 and sma50 < sma200:
        regime = "recovery attempt inside a longer downtrend (close above the 50-day)"
    else:
        regime = "trend losing momentum (the 50-day is flattening)"
    lines.append(f"- Trend regime: {regime}.")
    broke_down = range_state is not None and "BELOW" in range_state
    broke_up = range_state is not None and "ABOVE" in range_state
    if sma50 is not None and (broke_down or broke_up) and regime.startswith(("uptrend", "downtrend")):
        up = regime.startswith("uptrend")
        if up and broke_down:
            reading = ("a pullback inside the uptrend: bearish on a days horizon, while the uptrend "
                       f"holds above the 50-day ({sma50:.2f}).")
        elif not up and broke_up:
            reading = ("a bounce inside the downtrend: bullish on a days horizon, while the downtrend "
                       f"holds below the 50-day ({sma50:.2f}).")
        else:
            reading = f"a range break in the direction of the {'uptrend' if up else 'downtrend'}: continuation."
        lines.append(f"- Reading: {reading}")
    return lines


def _verified_rows(symbol: str, curr_date: str) -> pd.DataFrame:
    """OHLCV on or before curr_date, date-sorted. Raises if nothing usable.

    ``load_ohlcv`` already normalizes the Date column and filters out
    look-ahead rows, but we re-apply the cutoff defensively — this is a
    verification path, so it must not trust its input to be pre-filtered.
    """
    data = load_ohlcv(symbol, curr_date)
    if data is None or data.empty:
        raise ValueError(f"No OHLCV data available for {symbol}.")

    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df[df["Date"] <= pd.to_datetime(curr_date)].sort_values("Date")
    if df.empty:
        raise ValueError(f"No OHLCV rows on or before {curr_date} for {symbol}.")
    return df


def _fmt(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def build_verified_market_snapshot(
    symbol: str,
    curr_date: str,
    look_back_days: int = 30,
    indicators: Iterable[str] | None = None,
) -> str:
    """Render a ground-truth snapshot: latest OHLCV row, indicators, recent closes."""
    # `df` keeps the original capitalized OHLCV columns (Open/High/Low/Close/
    # Volume); stockstats `wrap()` lowercases columns and adds indicator
    # columns, so read raw prices from `df` and indicators from `stock_df`.
    df = _verified_rows(symbol, curr_date)
    stock_df = wrap(df.copy())
    for column in ("close_20_sma", "close_50_sma", "close_200_sma", "atr"):
        try:
            stock_df[column]  # triggers stockstats calculation for the structure section
        except Exception:  # noqa: BLE001 -- the section skips what it cannot compute
            pass

    selected = tuple(indicators or DEFAULT_SNAPSHOT_INDICATORS)
    indicator_values: dict[str, str] = {}
    for name in selected:
        needed = _required_rows(name)
        if needed > len(df):
            indicator_values[name] = f"N/A (needs {needed} rows, have {len(df)})"
            continue
        try:
            stock_df[name]  # triggers stockstats calculation
            indicator_values[name] = _fmt(stock_df.iloc[-1][name])
        except Exception as exc:  # noqa: BLE001 — one bad indicator shouldn't sink the snapshot
            indicator_values[name] = f"N/A ({type(exc).__name__})"

    latest = df.iloc[-1]
    latest_date = _fmt(latest["Date"])
    window = max(1, min(int(look_back_days), 30))
    recent = df.tail(window)

    lines = [
        f"## Verified market data snapshot for {symbol.upper()}",
        "",
        f"- Requested analysis date: {curr_date}",
        f"- Latest trading row used: {latest_date}",
        "- Rows after the requested analysis date are excluded before verification.",
        "",
        "### Latest verified OHLCV row",
        "",
        "| Field | Value |",
        "|---|---:|",
    ]
    for field in ("Open", "High", "Low", "Close", "Volume"):
        lines.append(f"| {field} | {_fmt(latest.get(field))} |")

    lines += ["", "### Verified technical indicators (latest row)", "",
              "| Indicator | Value |", "|---|---:|"]
    for name, value in indicator_values.items():
        lines.append(f"| {name} | {value} |")

    lines += ["", f"### Recent verified closes (last {len(recent)} rows)", "",
              "| Date | Close |", "|---|---:|"]
    for _, row in recent.iterrows():
        lines.append(f"| {_fmt(row['Date'])} | {_fmt(row.get('Close'))} |")

    lines += _structure_lines(df, stock_df)

    lines += [
        "",
        "Use this snapshot as the source of truth for exact OHLCV, price-level, "
        "and indicator-value claims. If another tool output conflicts with it, "
        "flag the discrepancy rather than inventing a reconciled number. Do not "
        "claim historical validation, support/resistance bounces, or exact "
        "percentage moves unless directly supported by tool output with concrete "
        "dates and prices. The 'Session & structure' figures are computed from "
        "these verified rows and may be cited as such.",
    ]
    return "\n".join(lines)
