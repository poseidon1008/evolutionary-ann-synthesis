"""Run GA, ES, PSO and NEAT-lite head-to-head on the same task — direct,
gradient-free synthesis of a digit classifier — with matching population
size and generation budgets, and write the results the dashboard reads.

    python src/compare_evolution.py
"""
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from evo_algorithms import (
    ESEvolver, GAEvolver, NeatLiteEvolver, PSOSwarm,
    fitness_flat, forward_flat, n_weights,
)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"
POP_SIZE = 120
GENERATIONS = 150
LAYER_SIZES = (64, 24, 10)  # fixed topology for GA / ES / PSO
SEED = 42


def run_fixed_topology(evolver_cls, name, X_train, y_train, X_val, y_val, n_in, n_out, **kwargs):
    evolver = evolver_cls(list(LAYER_SIZES), n_out, pop_size=POP_SIZE, seed=SEED, **kwargs)
    history = []
    t0 = time.time()
    for gen in range(GENERATIONS):
        best_fit, best_acc, mean_fit = evolver.step(X_train, y_train)
        history.append({"generation": gen, "best_val_fitness": best_fit, "best_val_acc_train": best_acc})
        if gen % 30 == 0 or gen == GENERATIONS - 1:
            print(f"  [{name}] gen {gen:3d}  best_fitness={best_fit:.4f}")
    elapsed = time.time() - t0
    best_genome = evolver.best(X_train, y_train)
    _, val_acc = fitness_flat(best_genome, list(LAYER_SIZES), X_val, y_val, n_out)
    return {
        "name": name, "history": history, "seconds": round(elapsed, 1),
        "val_accuracy": round(float(val_acc), 4),
        "params": n_weights(list(LAYER_SIZES)),
        "genome": best_genome,
    }


def run_neat_lite(X_train, y_train, X_val, y_val, n_in, n_out):
    evolver = NeatLiteEvolver(n_in, n_out, pop_size=POP_SIZE, seed=SEED)
    history = []
    t0 = time.time()
    for gen in range(GENERATIONS):
        best_fit, best_acc, mean_fit, n_hidden, n_conn = evolver.step(X_train, y_train, n_out)
        history.append({"generation": gen, "best_val_fitness": best_fit, "best_val_acc_train": best_acc,
                         "hidden_nodes": n_hidden, "connections": n_conn})
        if gen % 30 == 0 or gen == GENERATIONS - 1:
            print(f"  [NEAT-lite] gen {gen:3d}  best_fitness={best_fit:.4f}  "
                  f"hidden={n_hidden}  connections={n_conn}")
    elapsed = time.time() - t0
    best_genome = evolver.best(X_train, y_train, n_out)
    _, val_acc = evolver.fitness(best_genome, X_val, y_val, n_out)
    n_hidden, n_conn = evolver.genome_size(best_genome)
    return {
        "name": "NEAT-lite", "history": history, "seconds": round(elapsed, 1),
        "val_accuracy": round(float(val_acc), 4),
        "params": n_conn + n_hidden,  # connections (weights) + node biases-equivalent
        "hidden_nodes": n_hidden, "connections": n_conn,
        "evolver": evolver, "genome": best_genome,
    }


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    digits = load_digits()
    X, y = digits.data, digits.target
    n_in, n_out = X.shape[1], len(set(y))

    X_trainfull, X_test, y_trainfull, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainfull, y_trainfull, test_size=0.2, stratify=y_trainfull, random_state=42)
    # zero-mean/unit-variance inputs matter a lot for weight-scale-sensitive,
    # gradient-free search — same normalization for every algorithm
    mu, sigma = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    X_train, X_val, X_test = (X_train - mu) / sigma, (X_val - mu) / sigma, (X_test - mu) / sigma

    print(f"digits: train={len(X_train)} val={len(X_val)} test={len(X_test)}  "
          f"| pop={POP_SIZE} generations={GENERATIONS}\n")

    results = {}
    print("=== GA (crossover + mutation) ===")
    results["GA"] = run_fixed_topology(GAEvolver, "GA", X_train, y_train, X_val, y_val, n_in, n_out)

    print("\n=== ES (self-adaptive mutation, weighted recombination) ===")
    results["ES"] = run_fixed_topology(ESEvolver, "ES", X_train, y_train, X_val, y_val, n_in, n_out)

    print("\n=== PSO (swarm, no selection/reproduction) ===")
    results["PSO"] = run_fixed_topology(PSOSwarm, "PSO", X_train, y_train, X_val, y_val, n_in, n_out)

    print("\n=== NEAT-lite (topology + weight co-evolution) ===")
    neat_result = run_neat_lite(X_train, y_train, X_val, y_val, n_in, n_out)
    neat_evolver = neat_result.pop("evolver")
    results["NEAT-lite"] = neat_result

    # ---- final test-set accuracy for each winner ----
    print("\n=== Test-set accuracy ===")
    for name in ["GA", "ES", "PSO"]:
        _, acc = fitness_flat(results[name]["genome"], list(LAYER_SIZES), X_test, y_test, n_out)
        results[name]["test_accuracy"] = round(float(acc), 4)
        print(f"  {name:10s} test_acc={acc:.4f}  val_acc={results[name]['val_accuracy']:.4f}  "
              f"params={results[name]['params']}  time={results[name]['seconds']}s")
    _, acc = neat_evolver.fitness(results["NEAT-lite"]["genome"], X_test, y_test, n_out)
    results["NEAT-lite"]["test_accuracy"] = round(float(acc), 4)
    print(f"  {'NEAT-lite':10s} test_acc={acc:.4f}  val_acc={results['NEAT-lite']['val_accuracy']:.4f}  "
          f"params={results['NEAT-lite']['params']} (hidden={results['NEAT-lite']['hidden_nodes']}, "
          f"connections={results['NEAT-lite']['connections']})  time={results['NEAT-lite']['seconds']}s")

    # ---- figures ----
    plt.figure(figsize=(7, 4.5))
    for name, color in zip(["GA", "ES", "PSO", "NEAT-lite"], ["#2563eb", "#16a34a", "#d97706", "#7c3aed"]):
        h = results[name]["history"]
        plt.plot([r["generation"] for r in h], [r["best_val_acc_train"] for r in h],
                  label=name, color=color)
    plt.xlabel("generation")
    plt.ylabel("best-of-generation training accuracy")
    plt.title("Four gradient-free algorithms synthesizing the same classifier")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "evo_compare_convergence.png", dpi=140)
    plt.close()

    names = ["GA", "ES", "PSO", "NEAT-lite"]
    test_accs = [results[n]["test_accuracy"] for n in names]
    plt.figure(figsize=(6.5, 4.2))
    bars = plt.bar(names, test_accs, color=["#2563eb", "#16a34a", "#d97706", "#7c3aed"])
    for bar, acc in zip(bars, test_accs):
        plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.01, f"{acc:.3f}", ha="center")
    plt.ylim(0, 1.05)
    plt.ylabel("test accuracy")
    plt.title("Final test accuracy by algorithm")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "evo_compare_accuracy.png", dpi=140)
    plt.close()

    params = [results[n]["params"] for n in names]
    plt.figure(figsize=(6.5, 4.2))
    plt.bar(names, params, color=["#2563eb", "#16a34a", "#d97706", "#7c3aed"])
    plt.ylabel("weights (GA/ES/PSO) or connections+nodes (NEAT-lite)")
    plt.title("Network size by algorithm")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "evo_compare_size.png", dpi=140)
    plt.close()

    # ---- comparison.json for the app/README ----
    out = {
        "config": {"pop_size": POP_SIZE, "generations": GENERATIONS,
                    "fixed_layer_sizes": list(LAYER_SIZES)},
        "results": {
            n: {k: v for k, v in results[n].items() if k != "genome"}
            for n in names
        },
    }
    (ROOT / "reports" / "evo_comparison.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved comparison figures to {FIG_DIR} and reports/evo_comparison.json")


if __name__ == "__main__":
    main()
