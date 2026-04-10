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
    results["risk_distribution"].to_csv(output_dir / "risk_distribution.csv", index=False)
    results["triage_distribution"].to_csv(output_dir / "triage_distribution.csv", index=False)
    results["cohort_summary"].to_csv(output_dir / "cohort_summary.csv")

    lines = [
        "GlucoTrack Analytics - Model Summary",
        f"Dataset source: {source}",
        f"Rows loaded: {len(data)}",
        f"Best model: {results['best_model_name']}",
        f"High-risk patients: {results['operational_metrics']['high_risk_patients']}",
        f"Immediate review patients: {results['operational_metrics']['immediate_review_patients']}",
        "",
        "Leaderboard:",
        results["leaderboard"].to_string(index=False),
        "",
        "Top signals:",
        results["feature_importance"].to_string(index=False),
        "",
        "Risk distribution:",
        results["risk_distribution"].to_string(index=False),
        "",
        "Triage distribution:",
        results["triage_distribution"].to_string(index=False),
    ]
    (output_dir / "training_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
