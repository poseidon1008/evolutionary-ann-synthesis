"""Four population-based, gradient-free algorithms that synthesize a neural
network by direct search — no backpropagation anywhere in this file.

  - GAEvolver        genetic algorithm: tournament selection, uniform
                      crossover, Gaussian mutation, fixed topology
  - ESEvolver         (mu/mu, lambda) evolution strategy: weighted
                      recombination of the best mu, self-adaptive
                      per-individual mutation strength, no crossover
  - PSOSwarm          particle swarm optimization: velocity toward each
                      particle's own best and the swarm's best, no
                      selection/reproduction at all (not "evolutionary" in
                      the Darwinian sense, but the standard companion
                      algorithm in this space)
  - NeatLiteEvolver    topology *and* weight co-evolution: genomes start
                      minimal (inputs wired straight to outputs) and grow
                      hidden nodes/connections via structural mutation —
                      the only one of the four that also searches network
                      *shape*, not just weights for a shape you chose

GA/ES/PSO all optimize a flat weight vector for one fixed architecture, so
their results are directly comparable; NEAT-lite discovers its own
architecture, so its final network size is itself one of the results.

Fitness for all four is the same: negative mean softmax cross-entropy on a
validation split (a smooth, comparable signal), with accuracy reported
alongside for readability.
"""
import numpy as np

# ---------------------------------------------------------------------------
# shared: fixed-topology genome <-> weights, forward pass, loss
# ---------------------------------------------------------------------------

def layer_shapes(layer_sizes):
    return list(zip(layer_sizes, layer_sizes[1:]))


def n_weights(layer_sizes):
    return sum(a * b + b for a, b in layer_shapes(layer_sizes))


def unravel(flat, layer_sizes):
    """Flat weight vector -> list of (W, b) per layer."""
    layers, i = [], 0
    for a, b in layer_shapes(layer_sizes):
        w = flat[i:i + a * b].reshape(a, b); i += a * b
        bias = flat[i:i + b]; i += b
        layers.append((w, bias))
    return layers


def forward_flat(flat, layer_sizes, X):
    """Forward pass for a flat-vector genome. tanh hidden layers, linear
    output (softmax applied separately for loss/argmax for predictions)."""
    h = X
    layers = unravel(flat, layer_sizes)
    for w, b in layers[:-1]:
        h = np.tanh(h @ w + b)
    w, b = layers[-1]
    return h @ w + b  # logits


def softmax_xent(logits, y_int, n_classes):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    correct = probs[np.arange(len(y_int)), y_int]
    loss = -np.log(np.clip(correct, 1e-12, None)).mean()
    acc = (probs.argmax(axis=1) == y_int).mean()
    return loss, acc


def fitness_flat(flat, layer_sizes, X, y_int, n_classes):
    logits = forward_flat(flat, layer_sizes, X)
    loss, acc = softmax_xent(logits, y_int, n_classes)
    return -loss, acc


# ---------------------------------------------------------------------------
# GA — genetic algorithm (selection + crossover + mutation)
# ---------------------------------------------------------------------------

class GAEvolver:
    name = "GA"

    def __init__(self, layer_sizes, n_classes, pop_size=120, seed=0,
                 mutation_rate=0.2, mutation_sigma=0.2, tournament_k=3):
        self.layer_sizes = layer_sizes
        self.n_classes = n_classes
        self.pop_size = pop_size
        self.rng = np.random.default_rng(seed)
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.tournament_k = tournament_k
        d = n_weights(layer_sizes)
        self.population = self.rng.normal(0, 0.5, (pop_size, d))

    def _tournament(self, fits):
        idx = self.rng.choice(self.pop_size, self.tournament_k, replace=False)
        return self.population[idx[np.argmax(fits[idx])]]

    def step(self, X, y_int):
        results = [fitness_flat(g, self.layer_sizes, X, y_int, self.n_classes) for g in self.population]
        fits = np.array([r[0] for r in results])
        accs = np.array([r[1] for r in results])
        best_i = int(np.argmax(fits))
        elite = self.population[np.argsort(fits)[::-1][:2]]

        children = [elite[0], elite[1]]
        while len(children) < self.pop_size:
            pa, pb = self._tournament(fits), self._tournament(fits)
            # BLX-alpha blend crossover: each gene is drawn from the interval
            # around [pa, pb] (with a little extrapolation room) rather than
            # copied whole from one parent — a real-coded crossover suited to
            # continuous weight vectors, unlike a discrete gene-swap.
            lo, hi = np.minimum(pa, pb), np.maximum(pa, pb)
            span = (hi - lo) * 0.25
            child = self.rng.uniform(lo - span, hi + span)
            mut_mask = self.rng.random(child.shape) < self.mutation_rate
            child = child + mut_mask * self.rng.normal(0, self.mutation_sigma, child.shape)
            children.append(child)
        self.population = np.array(children)
        return fits[best_i], accs[best_i], fits.mean()

    def best(self, X, y_int):
        fits = np.array([fitness_flat(g, self.layer_sizes, X, y_int, self.n_classes)[0]
                          for g in self.population])
        return self.population[int(np.argmax(fits))]


# ---------------------------------------------------------------------------
# ES — (mu/mu, lambda) evolution strategy: weighted recombination + a single
# self-adaptive global mutation strength (sigma), no crossover between
# individuals in the GA sense.
# ---------------------------------------------------------------------------

class ESEvolver:
    name = "ES"

    def __init__(self, layer_sizes, n_classes, pop_size=120, seed=0, mu_frac=0.25):
        self.layer_sizes = layer_sizes
        self.n_classes = n_classes
        self.pop_size = pop_size
        self.mu = max(2, int(pop_size * mu_frac))
        self.rng = np.random.default_rng(seed)
        d = n_weights(layer_sizes)
        self.d = d
        self.mean = self.rng.normal(0, 0.3, d)
        self.sigma = 0.6
        self.tau = 1 / np.sqrt(2 * d)
        # weighted-recombination weights favour the fitter of the mu parents
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = w / w.sum()

    def step(self, X, y_int):
        offspring = self.mean + self.sigma * self.rng.normal(0, 1, (self.pop_size, self.d))
        results = [fitness_flat(g, self.layer_sizes, X, y_int, self.n_classes) for g in offspring]
        fits = np.array([r[0] for r in results])
        accs = np.array([r[1] for r in results])
        order = np.argsort(fits)[::-1][:self.mu]
        self.mean = self.weights @ offspring[order]
        # self-adaptive step size: shrink/grow based on a log-normal draw
        self.sigma = float(np.clip(self.sigma * np.exp(self.tau * self.rng.normal()), 1e-3, 3.0))
        best_i = order[0]
        return fits[best_i], accs[best_i], fits.mean()

    def best(self, X, y_int):
        return self.mean


# ---------------------------------------------------------------------------
# PSO — particle swarm optimization. Not "evolutionary" (no selection or
# reproduction — every particle survives every iteration and just moves),
# but the standard companion metaheuristic for this kind of comparison.
# ---------------------------------------------------------------------------

class PSOSwarm:
    name = "PSO"

    def __init__(self, layer_sizes, n_classes, pop_size=120, seed=0,
                 inertia=0.7, cognitive=1.5, social=1.5):
        self.layer_sizes = layer_sizes
        self.n_classes = n_classes
        self.pop_size = pop_size
        self.rng = np.random.default_rng(seed)
        self.inertia, self.cognitive, self.social = inertia, cognitive, social
        d = n_weights(layer_sizes)
        self.positions = self.rng.normal(0, 0.5, (pop_size, d))
        self.velocities = self.rng.normal(0, 0.1, (pop_size, d))
        self.pbest = self.positions.copy()
        self.pbest_fit = np.full(pop_size, -np.inf)
        self.gbest = self.positions[0].copy()
        self.gbest_fit = -np.inf

    def step(self, X, y_int):
        results = [fitness_flat(p, self.layer_sizes, X, y_int, self.n_classes) for p in self.positions]
        fits = np.array([r[0] for r in results])
        accs = np.array([r[1] for r in results])
        improved = fits > self.pbest_fit
        self.pbest[improved] = self.positions[improved]
        self.pbest_fit[improved] = fits[improved]

        gen_best_i = int(np.argmax(fits))
        if fits[gen_best_i] > self.gbest_fit:
            self.gbest_fit = fits[gen_best_i]
            self.gbest = self.positions[gen_best_i].copy()

        r1 = self.rng.random(self.positions.shape)
        r2 = self.rng.random(self.positions.shape)
        self.velocities = (self.inertia * self.velocities
                            + self.cognitive * r1 * (self.pbest - self.positions)
                            + self.social * r2 * (self.gbest - self.positions))
        self.positions = self.positions + self.velocities
        return self.gbest_fit, accs[gen_best_i], fits.mean()

    def best(self, X, y_int):
        return self.gbest


# ---------------------------------------------------------------------------
# NEAT-lite — topology AND weight co-evolution. Genomes start minimal
# (inputs wired straight to outputs, no hidden nodes) and grow via
# structural mutation: perturb a weight, add a connection, or split an
# existing connection with a new hidden node. Mutation-only (no crossover —
# matching genomes by innovation number for crossover is the most complex
# part of full NEAT; the "grows in complexity from a minimal start" property
# is the part of the algorithm this project cares about demonstrating).
#
# Each node gets a real-valued depth in [0, 1]; a new node inserted by
# splitting connection (u -> v) gets depth = midpoint(depth[u], depth[v]),
# which keeps the network feed-forward with no need for topological-sort
# bookkeeping — nodes are simply evaluated in ascending depth order.
# ---------------------------------------------------------------------------

class NeatLiteEvolver:
    name = "NEAT-lite"

    def __init__(self, n_in, n_out, pop_size=120, seed=0,
                 weight_mut_rate=0.5, weight_mut_sigma=0.15,
                 add_conn_rate=0.05, add_node_rate=0.03, initial_conn_prob=0.5):
        self.n_in, self.n_out = n_in, n_out
        self.pop_size = pop_size
        self.rng = np.random.default_rng(seed)
        self.weight_mut_rate = weight_mut_rate
        self.weight_mut_sigma = weight_mut_sigma
        self.add_conn_rate = add_conn_rate
        self.add_node_rate = add_node_rate
        self.population = [self._seed_genome(initial_conn_prob) for _ in range(pop_size)]

    def _seed_genome(self, conn_prob):
        """Minimal genome: inputs -> outputs, partially connected, no hidden nodes."""
        connections = []
        for i in range(self.n_in):
            for o in range(self.n_out):
                if self.rng.random() < conn_prob:
                    connections.append({
                        "in": ("in", i), "out": ("out", o),
                        "weight": float(self.rng.normal(0, 0.5)), "enabled": True,
                    })
        return {"hidden": [], "connections": connections}  # hidden: list of {"id","depth"}

    def _next_hidden_id(self, genome):
        return len(genome["hidden"])

    def mutate(self, genome):
        g = {"hidden": [dict(h) for h in genome["hidden"]],
             "connections": [dict(c) for c in genome["connections"]]}

        for c in g["connections"]:
            if self.rng.random() < self.weight_mut_rate:
                c["weight"] += float(self.rng.normal(0, self.weight_mut_sigma))

        depths = {("in", i): 0.0 for i in range(self.n_in)}
        depths.update({("out", o): 1.0 for o in range(self.n_out)})
        depths.update({("hid", h["id"]): h["depth"] for h in g["hidden"]})

        if self.rng.random() < self.add_conn_rate and g["connections"]:
            nodes = list(depths.keys())
            for _ in range(5):  # a few tries to find a valid forward edge
                a, b = self.rng.choice(len(nodes), 2, replace=False)
                u, v = nodes[a], nodes[b]
                if depths[u] < depths[v]:
                    exists = any(c["in"] == u and c["out"] == v for c in g["connections"])
                    if not exists:
                        g["connections"].append({"in": u, "out": v,
                                                   "weight": float(self.rng.normal(0, 0.5)),
                                                   "enabled": True})
                    break

        enabled = [c for c in g["connections"] if c["enabled"]]
        if self.rng.random() < self.add_node_rate and enabled:
            c = enabled[int(self.rng.integers(len(enabled)))]
            c["enabled"] = False
            new_id = self._next_hidden_id(g)
            new_depth = (depths[c["in"]] + depths[c["out"]]) / 2
            g["hidden"].append({"id": new_id, "depth": new_depth})
            g["connections"].append({"in": c["in"], "out": ("hid", new_id),
                                       "weight": 1.0, "enabled": True})
            g["connections"].append({"in": ("hid", new_id), "out": c["out"],
                                       "weight": c["weight"], "enabled": True})
        return g

    def forward(self, genome, X):
        n = X.shape[0]
        activations = {("in", i): X[:, i] for i in range(self.n_in)}
        depths = {("hid", h["id"]): h["depth"] for h in genome["hidden"]}
        depths.update({("out", o): 1.0 for o in range(self.n_out)})
        order = sorted(depths, key=lambda k: depths[k])

        incoming = {}
        for c in genome["connections"]:
            if c["enabled"]:
                incoming.setdefault(c["out"], []).append(c)

        for node in order:
            total = np.zeros(n)
            for c in incoming.get(node, []):
                total = total + c["weight"] * activations[c["in"]]
            activations[node] = total if node[0] == "out" else np.tanh(total)

        logits = np.stack([activations[("out", o)] for o in range(self.n_out)], axis=1)
        return logits

    def fitness(self, genome, X, y_int, n_classes):
        logits = self.forward(genome, X)
        loss, acc = softmax_xent(logits, y_int, n_classes)
        return -loss, acc

    def _tournament(self, fits, k=2):
        idx = self.rng.choice(self.pop_size, k, replace=False)
        return self.population[idx[np.argmax(fits[idx])]]

    def step(self, X, y_int, n_classes):
        results = [self.fitness(g, X, y_int, n_classes) for g in self.population]
        fits = np.array([r[0] for r in results])
        accs = np.array([r[1] for r in results])
        best_i = int(np.argmax(fits))
        ranked = [g for _, g in sorted(zip(fits, self.population), key=lambda p: p[0], reverse=True)]
        next_pop = [ranked[0]]  # elitism — light, so structural innovations get a chance to compete
        while len(next_pop) < self.pop_size:
            parent = self._tournament(fits)
            next_pop.append(self.mutate(parent))
        self.population = next_pop
        n_conn = sum(1 for c in self.population[0]["connections"] if c["enabled"])
        return fits[best_i], accs[best_i], fits.mean(), len(self.population[0]["hidden"]), n_conn

    def best(self, X, y_int, n_classes):
        fits = np.array([self.fitness(g, X, y_int, n_classes)[0] for g in self.population])
        return self.population[int(np.argmax(fits))]

    def genome_size(self, genome):
        n_hidden = len(genome["hidden"])
        n_conn = sum(1 for c in genome["connections"] if c["enabled"])
        return n_hidden, n_conn
