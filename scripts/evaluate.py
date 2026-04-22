"""Run phase-6 benchmark metrics and persist a report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import evaluate_all


def main() -> int:
    parser = argparse.ArgumentParser(description="Run phase-6 evaluation metrics")
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory where metrics artifacts will be written",
    )
    args = parser.parse_args()

    summary = evaluate_all()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "phase6_metrics.json"
    markdown_path = output_dir / "phase6_metrics.md"

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(summary.to_dict(), json_file, indent=2)
        json_file.write("\n")

    markdown_path.write_text(summary.to_markdown(), encoding="utf-8")

    print(f"Wrote metrics JSON: {json_path}")
    print(f"Wrote metrics report: {markdown_path}")
    print(f"Overall passed: {summary.overall_passed}")

    for metric_name, metric in summary.metrics.items():
        print(
            f"- {metric_name}: value={metric.value:.4f}, "
            f"target={metric.target:.4f}, passed={metric.passed}"
        )

    return 0 if summary.overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
