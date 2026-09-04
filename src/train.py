"""Run the GA architecture search, retrain the winner, compare to a fixed
baseline MLP, and write the model + report figures used by the README/app.

    python src/train.py
"""
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ga_search import build_model, param_count, run_ga

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"
MODEL_DIR = ROOT / "models"
FINAL_MAX_ITER = 400
BASELINE_ARCH = (100,)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    digits = load_digits()
    X, y = digits.data, digits.target
    n_in, n_out = X.shape[1], len(set(y))

    X_trainfull, X_test, y_trainfull, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainfull, y_trainfull, test_size=0.2, stratify=y_trainfull, random_state=42)

    scaler = StandardScaler().fit(X_trainfull)
    X_train_s, X_val_s = scaler.transform(X_train), scaler.transform(X_val)
    X_trainfull_s, X_test_s = scaler.transform(X_trainfull), scaler.transform(X_test)

    print(f"digits dataset: {X.shape[0]} samples, {n_in} features, {n_out} classes")
    print(f"train={len(X_train)} val={len(X_val)} (search) | trainfull={len(X_trainfull)} test={len(X_test)} (final)\n")

    print("=== GA architecture search ===")
    t0 = time.time()
    best_genome, best_val_acc, history = run_ga(
        X_train_s, y_train, X_val_s, y_val, n_in, n_out,
        pop_size=16, generations=12, seed=42)
    search_seconds = time.time() - t0
    print(f"\nsearch finished in {search_seconds:.1f}s")
    print(f"best genome: {best_genome}  (val_acc={best_val_acc:.4f})\n")

    print("=== Retraining evolved architecture on full training set ===")
    evolved = build_model(best_genome, FINAL_MAX_ITER, seed=1)
    evolved.fit(X_trainfull_s, y_trainfull)
    evolved_pred = evolved.predict(X_test_s)
    evolved_acc = evolved.score(X_test_s, y_test)
    evolved_params = param_count(best_genome, n_in, n_out)
    print(f"evolved test accuracy: {evolved_acc:.4f}  ({evolved_params:,} params, "
          f"{len(best_genome['hidden_layers'])} hidden layer(s) {best_genome['hidden_layers']})")

    print("\n=== Training fixed baseline MLP (100,) for comparison ===")
    baseline_genome = {"hidden_layers": BASELINE_ARCH, "activation": "relu",
                        "alpha": 1e-4, "learning_rate_init": 1e-3}
    baseline = build_model(baseline_genome, FINAL_MAX_ITER, seed=1)
    baseline.fit(X_trainfull_s, y_trainfull)
    baseline_pred = baseline.predict(X_test_s)
    baseline_acc = baseline.score(X_test_s, y_test)
    baseline_params = param_count(baseline_genome, n_in, n_out)
    print(f"baseline test accuracy: {baseline_acc:.4f}  ({baseline_params:,} params)")

    # ---- persist models + scaler ----
    joblib.dump({"model": evolved, "scaler": scaler, "genome": best_genome},
                MODEL_DIR / "evolved_mlp.joblib")
    joblib.dump({"model": baseline, "scaler": scaler, "genome": baseline_genome},
                MODEL_DIR / "baseline_mlp.joblib")

    # ---- figures ----
    gens = [h["generation"] for h in history]
    plt.figure(figsize=(6, 4))
    plt.plot(gens, [h["best_val_acc"] for h in history], marker="o", label="best of generation")
    plt.plot(gens, [h["mean_val_acc"] for h in history], marker=".", linestyle="--", label="population mean")
    plt.xlabel("generation")
    plt.ylabel("validation accuracy")
    plt.title("GA architecture search — convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ga_convergence.png", dpi=140)
    plt.close()

    plt.figure(figsize=(5.5, 4))
    labels = ["Evolved\n" + str(best_genome["hidden_layers"]), "Baseline\n(100,)"]
    accs = [evolved_acc, baseline_acc]
    bars = plt.bar(labels, accs, color=["#2563eb", "#94a3b8"])
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.01, f"{acc:.3f}", ha="center")
    plt.ylim(0, 1.05)
    plt.ylabel("test accuracy")
    plt.title("Evolved architecture vs. fixed baseline")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "comparison_bar.png", dpi=140)
    plt.close()

    ConfusionMatrixDisplay.from_predictions(y_test, evolved_pred, cmap="Blues")
    plt.title("Evolved MLP — confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=140)
    plt.close()

    # ---- metrics.json for the README/app ----
    metrics = {
        "dataset": {"n_samples": int(X.shape[0]), "n_features": n_in, "n_classes": n_out},
        "search": {
            "pop_size": 16, "generations": 12, "seconds": round(search_seconds, 1),
            "history": history,
        },
        "evolved": {
            "genome": {**best_genome, "hidden_layers": list(best_genome["hidden_layers"])},
            "params": evolved_params,
            "val_acc_during_search": round(best_val_acc, 4),
            "test_accuracy": round(float(evolved_acc), 4),
            "classification_report": classification_report(y_test, evolved_pred, output_dict=True),
        },
        "baseline": {
            "genome": {**baseline_genome, "hidden_layers": list(baseline_genome["hidden_layers"])},
            "params": baseline_params,
            "test_accuracy": round(float(baseline_acc), 4),
            "classification_report": classification_report(y_test, baseline_pred, output_dict=True),
        },
    }
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved models to {MODEL_DIR}, figures to {FIG_DIR}, metrics.json to reports/.")


if __name__ == "__main__":
    main()
