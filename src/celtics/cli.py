"""`celtics` command-line entry point."""

import argparse
import sys
from datetime import date
from pathlib import Path

from celtics.config import N_SIMS, Paths
from celtics.data.cache import load_all, refresh
from celtics.data.fetch import FetchBlockedError
from celtics.data.validate import DataValidationError
from celtics.features.build import build_features
from celtics.pipeline import load_or_train, run_backtest, run_update


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="celtics", description="Celtics season prediction model")
    parser.add_argument(
        "--date", type=date.fromisoformat, default=None, help="treat this YYYY-MM-DD as today"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="refresh the local data cache")
    sub.add_parser("train", help="refit the model now")
    update = sub.add_parser("update", help="daily run: fetch, simulate, write site JSON")
    update.add_argument("--offline", action="store_true", help="use cached data only")
    update.add_argument("--force-train", action="store_true")
    update.add_argument("--sims", type=int, default=N_SIMS)
    backtest = sub.add_parser("backtest", help="walk-forward backtest -> site/data/backtest.json")
    backtest.add_argument("--offline", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = Paths(Path.cwd())
    today = args.date or date.today()
    try:
        if args.cmd == "fetch":
            games = refresh(paths.raw_dir, today)
            print(f"cached {len(games)} games")
        elif args.cmd == "train":
            load_or_train(build_features(load_all(paths.raw_dir)), paths, today, force=True)
            print("model trained")
        elif args.cmd == "update":
            pred = run_update(
                paths, today, offline=args.offline, force_train=args.force_train, n_sims=args.sims
            )
            s = pred["team_summary"]
            print(
                f"{s['team']} {s['wins']}-{s['losses']} | proj {s['proj_wins_median']:.0f} wins "
                f"(80%: {s['proj_wins_p10']:.0f}-{s['proj_wins_p90']:.0f}) | "
                f"playoffs {s['p_playoffs']:.0%}"
            )
        elif args.cmd == "backtest":
            report = run_backtest(paths, today, offline=args.offline)
            for name, m in report["models"].items():
                print(f"{name:10s} log_loss={m['log_loss']:.4f} brier={m['brier']:.4f} acc={m['accuracy']:.3f}")
            print(f"coverage={report['coverage']} chosen_team_sd={report['chosen_team_sd']}")
    except FetchBlockedError as exc:
        print(f"fetch blocked: {exc}", file=sys.stderr)
        return 2
    except DataValidationError as exc:
        print(f"data validation failed: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
