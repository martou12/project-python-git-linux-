import sys
from pathlib import Path

# Ensure project root on PYTHONPATH (so "import src" works when running from /scripts)
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.data_fetch import fetch_price_series, resample_price
from src.metrics import compute_metrics, open_close_return_24h, realized_vol
from src.quant_a_single_asset import build_single_asset_result


REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily report (Quant A) for a single asset (CoinGecko).")
    parser.add_argument("--asset", default="bitcoin", help="CoinGecko coin id (e.g., bitcoin, ethereum).")
    parser.add_argument("--vs", default="eur", choices=["eur", "usd"])
    parser.add_argument("--history_days", type=int, default=30)
    parser.add_argument("--periodicity", default="raw", choices=["raw", "5min", "15min", "1H", "4H", "1D"])
    parser.add_argument("--strategy", default="Momentum", choices=["Buy & Hold", "Momentum", "SMA Crossover"])
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--sma_short", type=int, default=10)
    parser.add_argument("--sma_long", type=int, default=30)
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
    )

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "module": "Quant A - Single Asset",
        "asset_id": args.asset,
        "base_currency": args.vs,
        "history_days": args.history_days,
        "periodicity": args.periodicity,
        "strategy": res.strategy_name,
        "params": res.params,
        "price_now": float(res.prices.iloc[-1]),
        "price_last_24h": {
            **open_close_return_24h(res.prices),
            **realized_vol(res.prices),
        },
        "strategy_metrics": compute_metrics(res.equity),
    }

    out_name = f"daily_report_quant_a_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{args.asset}_{args.vs}.json"
    out_path = REPORTS_DIR / out_name
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[OK] Quant A report written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
