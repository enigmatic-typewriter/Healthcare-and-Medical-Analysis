from __future__ import annotations

from pathlib import Path

from src.healthcare_analytics import load_default_dataset, run_training_workflow


def main():
    data, source = load_default_dataset()
    results = run_training_workflow(data)

    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)
    results["leaderboard"].to_csv(output_dir / "model_leaderboard.csv", index=False)
    results["feature_importance"].to_csv(output_dir / "feature_importance.csv", index=False)

    lines = [
        "GlucoTrack Analytics - Training Summary",
        f"Dataset source: {source}",
        f"Rows loaded: {len(data)}",
        f"Best model: {results['best_model_name']}",
        "",
        "Leaderboard:",
        results["leaderboard"].to_string(index=False),
        "",
        "Top signals:",
        results["feature_importance"].to_string(index=False),
    ]
    (output_dir / "training_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
