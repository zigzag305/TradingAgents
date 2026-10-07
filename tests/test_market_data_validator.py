"""Tests for the deterministic market-data verification snapshot (#830/#881)."""

from __future__ import annotations

import pandas as pd
import pytest

import tradingagents.dataflows.market_data_validator as validator


def _sample_ohlcv() -> pd.DataFrame:
    dates = pd.bdate_range("2026-04-01", "2026-05-20")
    closes = [100 + i for i in range(len(dates))]
    return pd.DataFrame({
        "Date": dates,
        "Open": [c - 0.5 for c in closes],
        "High": [c + 1.0 for c in closes],
        "Low": [c - 1.0 for c in closes],
        "Close": closes,
        "Volume": [1_000_000 + i for i in range(len(dates))],
    })


@pytest.mark.unit
class TestVerifiedSnapshot:
    def test_excludes_future_rows(self, monkeypatch):
        data = pd.concat([
            _sample_ohlcv(),
            pd.DataFrame({"Date": [pd.Timestamp("2026-06-01")], "Open": [999.0],
                          "High": [999.0], "Low": [999.0], "Close": [999.0], "Volume": [999]}),
        ], ignore_index=True)
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: data)

        snap = validator.build_verified_market_snapshot("COF", "2026-05-13")
        assert "Verified market data snapshot for COF" in snap
        assert "Requested analysis date: 2026-05-13" in snap
        assert "Latest trading row used: 2026-05-13" in snap
        assert "999.00" not in snap          # future row excluded
        assert "boll_lb" in snap             # indicators present

    def test_uses_previous_trading_day_when_date_is_weekend(self, monkeypatch):
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: _sample_ohlcv())
        # 2026-05-16 is a Saturday; latest row should be Fri 2026-05-15
        snap = validator.build_verified_market_snapshot("COF", "2026-05-16")
        assert "Latest trading row used: 2026-05-15" in snap
        assert "Recent verified closes" in snap

    def test_raises_when_no_rows_on_or_before_date(self, monkeypatch):
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: _sample_ohlcv())
        with pytest.raises(ValueError):
            validator.build_verified_market_snapshot("COF", "2020-01-01")

    def test_raises_on_empty_data(self, monkeypatch):
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: pd.DataFrame())
        with pytest.raises(ValueError):
            validator.build_verified_market_snapshot("COF", "2026-05-13")

    def test_look_back_window_capped_at_30(self, monkeypatch):
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: _sample_ohlcv())
        snap = validator.build_verified_market_snapshot("COF", "2026-05-20", look_back_days=999)
        # last-N closes table has at most 30 data rows
        close_rows = [ln for ln in snap.splitlines() if ln.startswith("| 2026-")]
        assert 0 < len(close_rows) <= 30


@pytest.mark.unit
class TestTool:
    def test_tool_delegates_to_builder(self, monkeypatch):
        from tradingagents.agents.utils.market_data_validation_tools import (
            get_verified_market_snapshot,
        )
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: _sample_ohlcv())
        out = get_verified_market_snapshot.invoke(
            {"symbol": "COF", "curr_date": "2026-05-20"}
        )
        assert "Verified market data snapshot for COF" in out


def _frame(closes, start="2025-01-01", spread=1.0) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(closes))
    opens = [closes[0]] + list(closes[:-1])
    return pd.DataFrame({
        "Date": dates,
        "Open": opens,
        "High": [max(o, c) + spread for o, c in zip(opens, closes)],
        "Low": [min(o, c) - spread for o, c in zip(opens, closes)],
        "Close": closes,
        "Volume": [1_000_000] * len(closes),
    })


RISE = [40 + 0.4 * i for i in range(200)]
FLAG = [120 + (2.5 if i % 2 else -0.5) for i in range(10)]   # prior 10-row low ~118.5


def _snapshot(monkeypatch, closes, today=None):
    frame = _frame(closes)
    monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: frame)
    monkeypatch.setattr(validator, "_today_dates", lambda: {today} if today else set())
    last = frame["Date"].iloc[-1].strftime("%Y-%m-%d")
    return validator.build_verified_market_snapshot("SOL-USD", last), frame


@pytest.mark.unit
class TestSessionStructure:
    """The analyst is told not to assert % moves or range breaks without tool
    support; this section is that support, computed from the verified rows."""

    def test_breakdown_in_uptrend_is_stated_as_a_pullback(self, monkeypatch):
        snap, _ = _snapshot(monkeypatch, RISE + FLAG + [116.0])
        assert "### Session & structure" in snap
        assert "closed BELOW the prior 10-row low" in snap
        assert "x ATR(14)" in snap
        assert "Trend regime: uptrend" in snap
        assert "Reading: a pullback inside the uptrend: bearish on a days horizon" in snap

    def test_inside_range_day_reports_the_range_and_no_reading(self, monkeypatch):
        snap, _ = _snapshot(monkeypatch, RISE + FLAG + [120.5])
        assert "stayed inside the prior 10-row range" in snap
        assert "Reading:" not in snap

    def test_breakout_with_the_trend_is_continuation(self, monkeypatch):
        snap, _ = _snapshot(monkeypatch, RISE + FLAG + [126.0])
        assert "closed ABOVE the prior 10-row high" in snap
        assert "continuation" in snap

    def test_current_day_row_is_flagged_as_possibly_forming(self, monkeypatch):
        closes = RISE + FLAG + [116.0]
        last_day = _frame(closes)["Date"].iloc[-1].date()
        snap, _ = _snapshot(monkeypatch, closes, today=last_day)
        assert "may still be forming" in snap
        snap, _ = _snapshot(monkeypatch, closes, today=None)
        assert "may still be forming" not in snap

    def test_short_history_says_so_instead_of_guessing_a_trend(self, monkeypatch):
        snap, _ = _snapshot(monkeypatch, [100 + i for i in range(30)])
        assert "not enough history for a 50/200-day trend read" in snap

    def test_sideways_market_is_not_called_a_trend(self, monkeypatch):
        import math
        snap, _ = _snapshot(monkeypatch, [100 + 5 * math.sin(i / 6) for i in range(220)])
        assert "no trend: the 50-day and 200-day have converged" in snap

    def test_guidance_permits_citing_the_computed_section(self, monkeypatch):
        snap, _ = _snapshot(monkeypatch, RISE + FLAG + [116.0])
        assert "'Session & structure' figures are computed from these verified rows" in snap


@pytest.mark.unit
def test_indicator_table_refuses_partial_window_averages(monkeypatch):
    # stockstats fills short histories with a partial-window mean, which would
    # pass a 30-row average off as a verified 200-day SMA.
    snap, _ = _snapshot(monkeypatch, [100 + i for i in range(30)])
    assert "| close_200_sma | N/A (needs 200 rows, have 30) |" in snap
    assert "| close_50_sma | N/A (needs 50 rows, have 30) |" in snap
    assert "| close_10_ema | N/A" not in snap   # 30 rows is enough for a 10-row EMA


@pytest.mark.unit
def test_atr_scaling_is_omitted_without_enough_rows(monkeypatch):
    snap, _ = _snapshot(monkeypatch, [100 + i for i in range(10)])
    assert "| atr | N/A (needs 15 rows, have 10) |" in snap
    assert "x ATR(14)" not in snap
    assert "Latest session" in snap   # the % move is still reported
