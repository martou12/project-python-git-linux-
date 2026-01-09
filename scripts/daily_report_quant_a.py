import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH (so "import src" works when running from /scripts)
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import json
from datetime import datetime
from pathlib import Path as P

from src.metrics import compute_metrics, open_close_return_24h, realized_vol
from src.quant_a_single_asset import build_single_asset_result

REPORTS_DIR = P("reports")
REPORTS_DIR.mkdir(exist_ok=True)

LOGS_DIR = P("logs")
LOGS_DIR.mkdir(exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Daily report (Quant A) for a single asset using CoinGecko data."
    )
    parser.add_argument("--asset", default="bitcoin", help="CoinGecko coin id (e.g., bitcoin, ethereum, solana).")
    parser.add_argument("--vs", default="eur", choices=["eur", "usd"], help="Quote currency.")
    parser.add_argument("--history_days", type=int, default=30, help="Number of days of history to fetch.")
    parser.add_argument(
        "--periodicity",
        default="raw",
        choices=["raw", "5min", "15min", "1H", "4H", "1D"],
        help="Resampling rule. raw = native CoinGecko sampling.",
    )

    parser.add_argument(
        "--strategy",
        default="Momentum",
        choices=["Buy & Hold", "Momentum", "SMA Crossover"],
        help="Trading strategy.",
    )
    parser.add_argument("--lookback", type=int, default=20, help="Lookback window (#points) for Momentum.")
    parser.add_argument("--sma_short", type=int, default=10, help="Short SMA window (#points) for SMA Crossover.")
    parser.add_argument("--sma_long", type=int, default=30, help="Long SMA window (#points) for SMA Crossover.")

    parser.add_argument("--allow_short", action="store_true", help="Enable long/short positions when supported.")
    parser.add_argument("--leverage", type=float, default=1.0, help="Leverage multiplier applied to strategy returns.")
    parser.add_argument("--fee_bps", type=float, default=5.0, help="Transaction fees in basis points (bps).")
    parser.add_argument("--slippage_bps", type=float, default=5.0, help="Slippage in basis points (bps).")
    parser.add_argument("--rf", type=float, default=0.0, help="Annual risk-free rate (e.g., 0.02).")

    args = parser.parse_args()

    res = build_single_asset_result(
        asset_id=args.asset,
        vs=args.vs,
        days=args.history_days,
        periodicity=args.periodicity,
        strategy=args.strategy,
        lookback=args.lookback,
        sma_short=args.sma_short,
        sma_long=args.sma_long,
        allow_short=args.allow_short,
        leverage=args.leverage,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "module": "Quant A - Single Asset",
        "asset_id": args.asset,
        "quote_currency": args.vs,
        "history_days": args.history_days,
        "periodicity": args.periodicity,
        "strategy": res.strategy_name,
        "params": res.params,
        "meta": {
            k: (str(v) if k in {"cache_path"} else v)
            for k, v in res.meta.items()
            if k not in {"sma_short", "sma_long"}
        },
        "price_now": float(res.prices.iloc[-1]),
        "last_24h_price_stats": {**open_close_return_24h(res.prices), **realized_vol(res.prices)},
        "strategy_metrics": compute_metrics(res.equity, rf_annual=float(args.rf)),
    }

    out_name = (
        f"daily_report_quant_a_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{args.asset}_{args.vs}.json"
    )
    out_path = REPORTS_DIR / out_name
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[OK] Quant A report written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
