"""Instant-fill replay of one HK session.

Default race: V2 (782-dim) vs V3 family (1082-dim).
V2-only (pick date + zip, then optional extra zips):

    python -m src.replay_day --family v2 --interactive
    python -m src.replay_day --family v2 --date 2026-08-24 --interactive
    python -m src.replay_day --family v2 --date 2026-08-24 --zip models/news_gpu_v2/checkpoint_2026-08-12.zip --zip models/news_gpu_v2/best_model.zip

V2 best and ``checkpoint_2026-08-12.zip`` share ``trading_env.py``.
V3 / V3.1 use ``trading_env_v3.py`` (can short here). V3.2 uses
``trading_env_v3_2.py`` (long-only). Same bars, flat 1M, instant fills.
HK names trade; US stays stale. No orders, no pickle writes.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.data_loader import ENHANCED_PARQUET, overlay_live_ohlcv
from src.data_loader_v3 import ENHANCED_V3_PARQUET
from src.inference import FutuPaperBroker, _hk_now
from src.inference_v3 import _history_klines_v3
from src.trading_env import MAX_LEVERAGE, N_CORE, TradingEnv as TradingEnvV2
from src.trading_env_v3 import TradingEnv as TradingEnvV3
from src.trading_env_v3_2 import TradingEnv as TradingEnvV32
from src.utils import (
    CORE_TICKERS,
    DATA_LOGS,
    HK_TZ,
    INITIAL_CASH,
    MODELS_DIR,
    PROJECT_ROOT,
    TICKER_NAMES,
    is_ticker_market_open,
    setup_logging,
)

logger = setup_logging("airaire.replay_day")


def _first_existing(cands: list[Path]) -> Path | None:
    for path in cands:
        if path.exists():
            return path
    return None


def _is_stop_reply(text: str) -> bool:
    return text.strip().strip('"').strip("'").lower() in {"", "n", "no", "none"}


def resolve_zip(text: str) -> Path:
    raw = text.strip().strip('"').strip("'")
    if not raw:
        raise FileNotFoundError("Empty zip path.")
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Zip not found: {path}")
    return path


def list_v2_zips(limit: int = 24) -> list[Path]:
    folders = [
        MODELS_DIR / "news_gpu_v2",
        MODELS_DIR / "news_gpu_v2_20260823135426",
        MODELS_DIR / "old" / "news_gpu_v2_old",
    ]
    seen: set[str] = set()
    out: list[Path] = []
    preferred = (
        "checkpoint_2026-08-12.zip",
        "best_model.zip",
        "finetuned_2026-08-22.zip",
        "checkpoint_2026-08-18.zip",
        "checkpoint_2026-08-21.zip",
    )
    for folder in folders:
        if not folder.is_dir():
            continue
        ranked: list[Path] = []
        rest: list[Path] = []
        by_name = {p.name: p for p in folder.glob("*.zip")}
        for name in preferred:
            if name in by_name:
                ranked.append(by_name[name])
        rest = sorted(p for p in folder.glob("*.zip") if p.name not in preferred)
        for path in ranked + rest:
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            out.append(path)
            if len(out) >= limit:
                return out
    return out


def contestants_from_zips(zips: list[Path], family: str) -> list[dict[str, Any]]:
    used: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for zip_path in zips:
        stem = zip_path.stem
        n = used.get(stem, 0) + 1
        used[stem] = n
        label = stem if n == 1 else f"{stem} ({n})"
        rows.append({"id": label.lower().replace(" ", "_"), "label": label, "family": family, "zip": zip_path})
    return rows


def prompt_date(default: date | None = None) -> date:
    hint = (default or _hk_now().date()).isoformat()
    raw = input(f"Date YYYY-MM-DD [Enter = {hint}]: ").strip()
    if not raw:
        return default or _hk_now().date()
    return date.fromisoformat(raw)


def prompt_zip_paths(family: str) -> list[Path]:
    catalog = list_v2_zips() if family == "v2" else []
    print("")
    print(f"Env family: {family}  ({'V2 trading_env.py 782-dim' if family == 'v2' else family})")
    if catalog:
        print("Known V2 zips:")
        for i, path in enumerate(catalog, start=1):
            try:
                rel = path.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = path
            print(f"  [{i:>2}] {rel}")
        print("Pick a number, or paste a .zip path.")
    else:
        print("Paste a .zip path (relative to the repo or absolute).")
    zips: list[Path] = []
    first = True
    while True:
        if first:
            raw = input("Model: ").strip()
            if _is_stop_reply(raw):
                raise SystemExit("Need at least one model zip.")
        else:
            raw = input("Another model to compare? Path, number, N, or Enter to run: ").strip()
            if _is_stop_reply(raw):
                break
        picked: Path | None = None
        if catalog and raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(catalog):
                picked = catalog[idx - 1]
            else:
                print(f"  Number must be 1–{len(catalog)}.")
                continue
        else:
            try:
                picked = resolve_zip(raw)
            except FileNotFoundError as exc:
                print(f"  {exc}")
                continue
        zips.append(picked)
        print(f"  + {picked}")
        first = False
    return zips


def default_contestants() -> list[dict[str, Any]]:
    v2_dir = MODELS_DIR / "news_gpu_v2_20260823135426"
    return [
        {
            "id": "v2_0812",
            "label": "V2 08-12",
            "family": "v2",
            "zip": _first_existing(
                [
                    v2_dir / "checkpoint_2026-08-12.zip",
                    MODELS_DIR / "news_gpu_v2" / "checkpoint_2026-08-12.zip",
                    MODELS_DIR / "old" / "news_gpu_v2_old" / "checkpoint_2026-08-12.zip",
                ]
            ),
        },
        {
            "id": "v2_best",
            "label": "V2 best",
            "family": "v2",
            "zip": _first_existing(
                [
                    MODELS_DIR / "news_gpu_v2" / "finetuned_2026-08-22.zip",
                    v2_dir / "finetuned_2026-08-22.zip",
                    MODELS_DIR / "news_gpu_v2" / "best_model.zip",
                    v2_dir / "best_model.zip",
                    MODELS_DIR / "news_gpu_v2" / "checkpoint_2026-08-21.zip",
                    v2_dir / "checkpoint_2026-08-21.zip",
                    MODELS_DIR / "old" / "news_gpu_v2_old" / "best_model.zip",
                ]
            ),
        },
        {
            "id": "v3",
            "label": "V3",
            "family": "v3",
            "zip": _first_existing(
                [
                    MODELS_DIR / "news_gpu_v3" / "best_model.zip",
                    MODELS_DIR / "old" / "news_gpu_v3" / "best_model.zip",
                ]
            ),
        },
        {
            "id": "v3_1",
            "label": "V3.1",
            "family": "v3",
            "zip": _first_existing([MODELS_DIR / "news_gpu_v3_1" / "best_model.zip"]),
        },
        {
            "id": "v3_2",
            "label": "V3.2",
            "family": "v3_2",
            "zip": _first_existing(
                [
                    MODELS_DIR / "news_gpu_v3_2" / "best_model.zip",
                    MODELS_DIR / "news_gpu_v3_2" / "checkpoint_2026-07-31.zip",
                ]
            ),
        },
    ]


def _aware_hk(ts: Any) -> datetime:
    raw = pd.Timestamp(ts).to_pydatetime()
    if raw.tzinfo is None:
        return raw.replace(tzinfo=HK_TZ)
    return raw.astimezone(HK_TZ)


def _gate_keep_closed(action: np.ndarray, env: TradingEnvV2, bar_dt: Any) -> np.ndarray:
    """Open names get the policy; closed names keep current weight (no stale US fill)."""
    prices = env._current_closes()
    equity = max(float(env._mark_to_market(prices)), 1.0)
    out = np.asarray(action, dtype=np.float64).reshape(-1).copy()
    if out.shape[0] != N_CORE:
        out = np.zeros(N_CORE, dtype=np.float64)
    aware = _aware_hk(bar_dt)
    for i, ticker in enumerate(CORE_TICKERS):
        if is_ticker_market_open(ticker, aware):
            continue
        out[i] = (float(env._holdings[i]) * float(prices[i])) / equity
    abs_sum = float(np.abs(out).sum())
    if abs_sum > MAX_LEVERAGE:
        out = out * (MAX_LEVERAGE / abs_sum)
    return out.astype(np.float32)


def last_news_scores(panel: pd.DataFrame, session: date) -> dict[str, float]:
    """Last parquet news_score at or before this session. Replay does not call Alpha Vantage."""
    scores = {t: 0.0 for t in CORE_TICKERS}
    if panel is None or panel.empty or "news_score" not in panel.columns:
        return scores
    df = panel.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    cutoff = pd.Timestamp(session) + pd.Timedelta(days=1)
    for ticker in CORE_TICKERS:
        part = df.loc[(df["ticker"].astype(str) == ticker) & (df["datetime"] < cutoff)].sort_values("datetime")
        if part.empty:
            continue
        val = pd.to_numeric(part["news_score"], errors="coerce").dropna()
        if not val.empty:
            scores[ticker] = float(np.clip(val.iloc[-1], -1.0, 1.0))
    return scores


def fetch_live_klines(broker: FutuPaperBroker, start: datetime, end: datetime) -> pd.DataFrame:
    live = _history_klines_v3(broker, start, end)
    if live is None or live.empty:
        logger.warning("OpenD returned no klines. Replay will use parquet only (likely no today).")
        return pd.DataFrame(columns=["datetime", "ticker", "open", "high", "low", "close", "volume"])
    return live


def build_panel(parquet: Path, live: pd.DataFrame, now: datetime) -> pd.DataFrame:
    if not parquet.exists():
        raise FileNotFoundError(f"Missing panel {parquet}")
    panel = pd.read_parquet(parquet)
    return overlay_live_ohlcv(panel, live, now=now.replace(tzinfo=None) if now.tzinfo else now)


def _env_for(family: str):
    if family == "v2":
        return TradingEnvV2
    if family == "v3_2":
        return TradingEnvV32
    return TradingEnvV3


def _holdings_dict(env: Any) -> dict[str, float]:
    return {t: float(env._holdings[i]) for i, t in enumerate(CORE_TICKERS)}


def replay_one(
    *,
    label: str,
    family: str,
    zip_path: Path,
    panel: pd.DataFrame,
    session: date,
    initial_cash: float,
    news_scores: dict[str, float],
) -> dict[str, Any]:
    env_cls = _env_for(family)
    env = env_cls(df=panel, initial_cash=initial_cash, window_days=30, news_scores=news_scores)
    model = PPO.load(str(zip_path), device="cpu")
    need = int(np.prod(env.observation_space.shape))
    got = int(np.prod(model.observation_space.shape))
    if got != need:
        raise RuntimeError(
            f"{label}: zip is {got}-dim but {family} env is {need}-dim. "
            "V2 zips need --family v2. V3/V3.1 need --family v3. V3.2 needs --family v3_2."
        )
    dts = list(env.datetimes)
    today_idx = [i for i, ts in enumerate(dts) if pd.Timestamp(ts).date() == session]
    if not today_idx:
        last = pd.Timestamp(dts[-1]).date() if dts else None
        raise RuntimeError(
            f"{label}: no bars dated {session} in the panel (last={last}). "
            "OpenD must be up so today's HK 10-min klines overlay."
        )

    env.reset()
    env.seek_to_datetime(dts[today_idx[0]])
    env.restore_portfolio(initial_cash, {t: 0.0 for t in CORE_TICKERS})
    env.set_news_scores(news_scores)

    us_last = {}
    for ticker in ("US.COST", "US.KO", "US.SPX"):
        part = panel.loc[panel["ticker"].astype(str) == ticker]
        if part.empty:
            us_last[ticker] = None
        else:
            us_last[ticker] = str(pd.to_datetime(part["datetime"]).max())

    curve: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    prev_h = _holdings_dict(env)

    while True:
        bar_dt = env._current_dt()
        if pd.Timestamp(bar_dt).date() != session:
            break
        obs = env._get_obs()
        raw, _ = model.predict(obs, deterministic=True)
        raw = np.nan_to_num(np.asarray(raw, dtype=np.float32).reshape(-1), nan=0.0)
        gated = _gate_keep_closed(raw, env, bar_dt)
        if family == "v3_2":
            gated = np.clip(gated, 0.0, 1.0).astype(np.float32)
        _obs, _rew, terminated, truncated, info = env.step(gated)
        now_h = {t: float((info.get("holdings") or {}).get(t, 0.0)) for t in CORE_TICKERS}
        for ticker in CORE_TICKERS:
            delta = now_h[ticker] - prev_h[ticker]
            if abs(delta) < 1e-6:
                continue
            trades.append(
                {
                    "time": str(bar_dt),
                    "ticker": ticker,
                    "name": TICKER_NAMES.get(ticker, ticker),
                    "delta": float(delta),
                    "side": "BUY" if delta > 0 else "SELL",
                    "weight": float(gated[CORE_TICKERS.index(ticker)]),
                }
            )
        prev_h = now_h
        curve.append(
            {
                "time": str(bar_dt),
                "equity": float(info["equity"]),
                "cash": float(info["cash"]),
                "holdings": now_h,
            }
        )
        if terminated or truncated:
            break
        if pd.Timestamp(env._current_dt()).date() != session:
            break

    start_eq = float(initial_cash)
    end_eq = float(curve[-1]["equity"]) if curve else start_eq
    pnl = end_eq - start_eq
    return {
        "id": label.lower().replace(".", "_").replace(" ", "_"),
        "label": label,
        "family": family,
        "env": "v2 782-dim" if family == "v2" else ("v3.2 long-only 1082-dim" if family == "v3_2" else "v3 1082-dim"),
        "zip": str(zip_path),
        "bars": len(curve),
        "start_equity": start_eq,
        "end_equity": end_eq,
        "pnl": pnl,
        "return_pct": 100.0 * pnl / start_eq if start_eq else 0.0,
        "end_cash": float(curve[-1]["cash"]) if curve else start_eq,
        "end_holdings": curve[-1]["holdings"] if curve else {t: 0.0 for t in CORE_TICKERS},
        "us_last_bar": us_last,
        "news_scores": {t: float(news_scores.get(t, 0.0)) for t in CORE_TICKERS},
        "first_bar": curve[0]["time"] if curve else None,
        "last_bar": curve[-1]["time"] if curve else None,
        "trades": trades,
        "curve": curve,
    }


def run(
    *,
    session: date,
    skip_futu: bool,
    initial_cash: float,
    contestants: list[dict[str, Any]] | None = None,
    json_tag: str = "",
) -> dict[str, Any]:
    now = _hk_now()
    if contestants is None:
        all_c = default_contestants()
        skipped = [c["label"] for c in all_c if c["zip"] is None]
        if skipped:
            logger.warning("Skipping missing zips: %s", ", ".join(skipped))
        contestants = [c for c in all_c if c["zip"] is not None]
    if not contestants:
        raise FileNotFoundError("No contestant zips found.")

    live = pd.DataFrame()
    broker = FutuPaperBroker(dry_run=skip_futu)
    try:
        broker.connect()
        if not skip_futu and not broker.dry_run:
            start = datetime(session.year, session.month, session.day, 0, 0, 0)
            live = fetch_live_klines(broker, start, now.replace(tzinfo=None))
        elif skip_futu:
            logger.warning("--skip-futu: no live HK overlay.")
    finally:
        broker.close()

    v2_panel = build_panel(ENHANCED_PARQUET, live, now)
    need_v3 = any(str(row.get("family") or "") != "v2" for row in contestants)
    if need_v3:
        v3_panel = build_panel(ENHANCED_V3_PARQUET, live, now) if ENHANCED_V3_PARQUET.exists() else v2_panel
        if not ENHANCED_V3_PARQUET.exists():
            logger.warning("No enhanced_v3.parquet — V3/V3.1/V3.2 will use the V2 panel (no HSI/SPX volume).")
    else:
        v3_panel = v2_panel
    news_scores = last_news_scores(v2_panel if not v2_panel.empty else v3_panel, session)
    logger.info(
        "Replay news_scores (parquet ffill, not live Alpha Vantage): %s",
        {TICKER_NAMES.get(t, t): round(news_scores[t], 3) for t in CORE_TICKERS},
    )

    results = []
    for row in contestants:
        panel = v2_panel if row["family"] == "v2" else v3_panel
        logger.info("Replaying %s from %s", row["label"], row["zip"])
        results.append(
            replay_one(
                label=row["label"],
                family=row["family"],
                zip_path=Path(row["zip"]),
                panel=panel,
                session=session,
                initial_cash=initial_cash,
                news_scores=news_scores,
            )
        )

    ranked = sorted(results, key=lambda r: r["pnl"], reverse=True)
    payload = {
        "session": session.isoformat(),
        "clock_hkt": now.isoformat(),
        "initial_cash": initial_cash,
        "rules": (
            "Instant fill at each completed 10-min close. Flat 1M at HK open. "
            "Same parquet news_score for every model (no live Alpha Vantage). HK names only; US last bar is stale."
        ),
        "news_scores": news_scores,
        "winner": ranked[0]["label"] if ranked else None,
        "models": results,
    }
    DATA_LOGS.mkdir(parents=True, exist_ok=True)
    tag = json_tag.strip().strip("_")
    stamp = f"replay_{session.isoformat()}" + (f"_{tag}" if tag else "")
    out = DATA_LOGS / f"{stamp}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", out)

    print("")
    print(f"HK session {session}  instant-fill replay  start={initial_cash:,.0f} HKD")
    print(f"{'model':<28} {'env':<22} {'pnl':>12} {'return':>8} {'end equity':>14} {'bars':>6}")
    for row in ranked:
        print(
            f"{row['label']:<28} {row['env']:<22} {row['pnl']:>+12.2f} {row['return_pct']:>7.2f}% "
            f"{row['end_equity']:>14.2f} {row['bars']:>6}"
        )
        print(f"           {row['zip']}")
    print("News (last parquet score, frozen for the day — not today's Alpha Vantage):")
    print("  " + "  ".join(f"{TICKER_NAMES.get(t, t)}={news_scores[t]:+.3f}" for t in CORE_TICKERS))
    print("")
    print(f"Winner: {payload['winner']}")
    print("US bars are last completed US session (stale). HK is today's 10-min tape.")
    print(f"JSON: {out}")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Instant-fill HK-session replay")
    p.add_argument("--date", default="", help="YYYY-MM-DD (default: today's HKT date)")
    p.add_argument("--skip-futu", action="store_true", help="Do not pull OpenD klines.")
    p.add_argument("--cash", type=float, default=INITIAL_CASH)
    p.add_argument("--family", default=None, choices=("v2", "v3", "v3_2"), help="Env for --zip / --interactive (default v2).")
    p.add_argument("--zip", action="append", dest="zips", default=None, help="PPO zip. Repeat for a comparison.")
    p.add_argument("--interactive", action="store_true", help="Ask for date (unless --date) and model paths.")
    p.add_argument("--tag", default="", help="Suffix on replay_YYYY-MM-DD_<tag>.json")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    session = date.fromisoformat(args.date.strip()) if args.date.strip() else None
    family = args.family
    contestants: list[dict[str, Any]] | None = None
    tag = args.tag.strip()

    if args.interactive:
        family = family or "v2"
        if session is None:
            session = prompt_date()
        else:
            print(f"Date locked: {session.isoformat()}")
        zips = prompt_zip_paths(family)
        contestants = contestants_from_zips(zips, family)
        tag = tag or family
    elif args.zips:
        family = family or "v2"
        if session is None:
            session = _hk_now().date()
        contestants = contestants_from_zips([resolve_zip(z) for z in args.zips], family)
        tag = tag or family
    elif session is None:
        session = _hk_now().date()

    run(
        session=session,
        skip_futu=args.skip_futu,
        initial_cash=float(args.cash),
        contestants=contestants,
        json_tag=tag,
    )


if __name__ == "__main__":
    main()
