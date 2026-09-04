"""Bonus demo: evolve a network's *weights* directly with a GA — no
backpropagation anywhere. ga_search.py evolves architecture/hyperparameters
and still trains each candidate with ordinary gradient descent; this script
is the more literal "evolve an ANN" — a fixed 2-4-1 MLP whose weight vector
is the genome, fitness is classification accuracy on XOR, and evolution
(selection + crossover + Gaussian mutation) is the only optimizer.

    python src/neuroevolution_xor.py
"""
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARCH = (2, 4, 1)  # inputs -> hidden -> output
N_WEIGHTS = sum(a * b + b for a, b in zip(ARCH, ARCH[1:]))

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([0, 1, 1, 0], dtype=float)  # XOR


def unpack(genome):
    """Flat weight vector -> (W1, b1, W2, b2) for the fixed 2-4-1 MLP."""
    n_in, n_hidden, n_out = ARCH
    i = 0
    w1 = genome[i:i + n_in * n_hidden].reshape(n_in, n_hidden); i += n_in * n_hidden
    b1 = genome[i:i + n_hidden]; i += n_hidden
    w2 = genome[i:i + n_hidden * n_out].reshape(n_hidden, n_out); i += n_hidden * n_out
    b2 = genome[i:i + n_out]
    return w1, b1, w2, b2


def forward(genome, X):
    w1, b1, w2, b2 = unpack(genome)
    h = np.tanh(X @ w1 + b1)
    out = 1 / (1 + np.exp(-(h @ w2 + b2)))  # sigmoid
    return out.ravel()


def fitness(genome):
    pred = forward(genome, X)
    correct = ((pred > 0.5).astype(float) == Y).mean()
    mse = float(np.mean((pred - Y) ** 2))
    return correct - 0.05 * mse  # accuracy, MSE as a tiebreaker


def run(pop_size=60, generations=80, seed=7, log=print):
    rng = np.random.default_rng(seed)
    population = [rng.normal(0, 1, N_WEIGHTS) for _ in range(pop_size)]
    history = []

    for gen in range(generations):
        fits = [fitness(g) for g in population]
        best_i = int(np.argmax(fits))
        history.append({"generation": gen, "best_fitness": fits[best_i]})
        if gen % 10 == 0 or gen == generations - 1:
            log(f"gen {gen:3d}  best_fitness={fits[best_i]:.4f}")

        ranked = [g for g, _ in sorted(zip(population, fits), key=lambda p: p[1], reverse=True)]
        next_pop = ranked[:4]  # elitism
        while len(next_pop) < pop_size:
            a, b = rng.choice(ranked[:pop_size // 2], size=2, replace=False)
            mask = rng.random(N_WEIGHTS) < 0.5
            child = np.where(mask, a, b)
            if rng.random() < 0.8:
                child = child + rng.normal(0, 0.4, N_WEIGHTS) * (rng.random(N_WEIGHTS) < 0.2)
            next_pop.append(child)
        population = next_pop

    best = ranked[0]
    return best, history


if __name__ == "__main__":
    best, history = run()
    pred = forward(best, X)
    print("\nfinal predictions on XOR:")
    for row, p, target in zip(X, pred, Y):
        print(f"  {row.tolist()} -> {p:.3f} (target {int(target)})")
    acc = ((pred > 0.5).astype(float) == Y).mean()
    print(f"\naccuracy: {acc:.0%}  ({N_WEIGHTS} weights evolved, zero gradient steps)")

    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot([h["generation"] for h in history], [h["best_fitness"] for h in history])
    plt.xlabel("generation")
    plt.ylabel("best fitness")
    plt.title("Pure weight-evolution on XOR (no backprop)")
    plt.tight_layout()
    plt.savefig(fig_dir / "xor_convergence.png", dpi=140)
    plt.close()
