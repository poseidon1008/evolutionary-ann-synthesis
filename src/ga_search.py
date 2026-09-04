"""Genetic algorithm that searches MLP architectures/hyperparameters.

Each "genome" encodes a small feed-forward network config (1-3 hidden
layers, activation, L2 strength, learning rate). A genome's fitness is the
accuracy of an MLP trained with that config on a held-out validation split,
minus a small penalty on parameter count so the search mildly favors
compact networks when accuracy ties. Selection/crossover/mutation operate
on the genome dict directly — training itself is still ordinary backprop
(via scikit-learn); the GA is doing architecture/hyperparameter search
(neuroevolution of *design*, not of the weights themselves — see
neuroevolution_xor.py for the weight-evolution version).
"""
import random

from sklearn.neural_network import MLPClassifier

LAYER_SIZES = [8, 16, 32, 64, 128]
ACTIVATIONS = ["relu", "tanh", "logistic"]
ALPHA_RANGE = (-5, -1)          # log10 bounds
LR_RANGE = (-4, -1)             # log10 bounds
SEARCH_MAX_ITER = 150           # cheap-but-representative training budget used during search


def random_genome(rng):
    n_layers = rng.randint(1, 3)
    return {
        "hidden_layers": tuple(rng.choice(LAYER_SIZES) for _ in range(n_layers)),
        "activation": rng.choice(ACTIVATIONS),
        "alpha": 10 ** rng.uniform(*ALPHA_RANGE),
        "learning_rate_init": 10 ** rng.uniform(*LR_RANGE),
    }


def param_count(genome, n_in, n_out):
    """Total weights+biases an MLP with this genome would have."""
    sizes = [n_in, *genome["hidden_layers"], n_out]
    return sum(a * b + b for a, b in zip(sizes, sizes[1:]))


def build_model(genome, max_iter, seed):
    return MLPClassifier(
        hidden_layer_sizes=genome["hidden_layers"],
        activation=genome["activation"],
        alpha=genome["alpha"],
        learning_rate_init=genome["learning_rate_init"],
        solver="adam",
        max_iter=max_iter,
        random_state=seed,
    )


def fitness(genome, X_train, y_train, X_val, y_val, n_in, n_out, size_penalty=1.2e-6):
    model = build_model(genome, SEARCH_MAX_ITER, seed=0)
    model.fit(X_train, y_train)
    acc = model.score(X_val, y_val)
    penalty = size_penalty * param_count(genome, n_in, n_out)
    return acc - penalty, acc


def tournament_select(pop, scored, rng, k=3):
    contestants = rng.sample(list(zip(pop, scored)), k)
    return max(contestants, key=lambda pair: pair[1])[0]


def crossover(parent_a, parent_b, rng):
    """Gene-wise uniform crossover — each gene comes from one parent, so
    every child is a valid genome (no size mismatches to repair)."""
    return {gene: (parent_a[gene] if rng.random() < 0.5 else parent_b[gene])
            for gene in parent_a}


def mutate(genome, rng, rate=0.3):
    g = dict(genome)
    if rng.random() < rate:
        layers = list(g["hidden_layers"])
        move = rng.choice(["resize", "add", "remove"]) if len(layers) > 1 else rng.choice(["resize", "add"])
        if move == "resize":
            i = rng.randrange(len(layers))
            layers[i] = rng.choice(LAYER_SIZES)
        elif move == "add" and len(layers) < 3:
            layers.append(rng.choice(LAYER_SIZES))
        elif move == "remove" and len(layers) > 1:
            layers.pop(rng.randrange(len(layers)))
        g["hidden_layers"] = tuple(layers)
    if rng.random() < rate:
        g["activation"] = rng.choice(ACTIVATIONS)
    if rng.random() < rate:
        g["alpha"] = min(10 ** ALPHA_RANGE[1], max(10 ** ALPHA_RANGE[0], g["alpha"] * rng.uniform(0.3, 3)))
    if rng.random() < rate:
        g["learning_rate_init"] = min(10 ** LR_RANGE[1], max(10 ** LR_RANGE[0],
                                       g["learning_rate_init"] * rng.uniform(0.3, 3)))
    return g


def run_ga(X_train, y_train, X_val, y_val, n_in, n_out,
           pop_size=16, generations=12, elitism=2, seed=42, log=print):
    rng = random.Random(seed)
    population = [random_genome(rng) for _ in range(pop_size)]
    history = []
    best_genome, best_fit, best_acc = None, -1, -1

    for gen in range(generations):
        results = [fitness(g, X_train, y_train, X_val, y_val, n_in, n_out) for g in population]
        fits = [r[0] for r in results]
        accs = [r[1] for r in results]

        gen_best_i = max(range(pop_size), key=lambda i: fits[i])
        if fits[gen_best_i] > best_fit:
            best_fit, best_acc, best_genome = fits[gen_best_i], accs[gen_best_i], population[gen_best_i]

        history.append({
            "generation": gen,
            "best_fitness": fits[gen_best_i],
            "best_val_acc": accs[gen_best_i],
            "mean_val_acc": sum(accs) / pop_size,
        })
        log(f"gen {gen:2d}  best_val_acc={accs[gen_best_i]:.4f}  "
            f"mean_val_acc={sum(accs)/pop_size:.4f}  best_arch={population[gen_best_i]['hidden_layers']}")

        ranked = sorted(zip(population, fits), key=lambda p: p[1], reverse=True)
        next_pop = [g for g, _ in ranked[:elitism]]
        while len(next_pop) < pop_size:
            parent_a = tournament_select(population, fits, rng)
            parent_b = tournament_select(population, fits, rng)
            child = mutate(crossover(parent_a, parent_b, rng), rng)
            next_pop.append(child)
        population = next_pop

    return best_genome, best_acc, history
