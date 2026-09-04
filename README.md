# Evolutionary ANN Synthesis

Two demonstrations of using **genetic algorithms to build neural networks
instead of (or alongside) gradient descent**:

1. **Architecture search** — a GA evolves an MLP's topology and
   hyperparameters (layer sizes, activation, L2 strength, learning rate)
   for a handwritten-digit classifier; each candidate is still trained by
   ordinary backprop, but the GA is what decides *what* to train.
2. **Pure weight evolution** — a small bonus script evolves a fixed
   network's *weights* directly with selection + crossover + mutation,
   solving XOR with **zero gradient steps**.

**Stack:** Python · scikit-learn · NumPy · Streamlit (custom from-scratch GA — no NEAT/DEAP dependency)

🔗 **Live demo:** _deployed on Streamlit Community Cloud_ — see [Deployment](#deployment)

---

## 1. Architecture search (neuroevolution of design)

### Problem

Find a compact, accurate MLP for the [scikit-learn `digits`
dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#digits-dataset)
(1,797 8×8 handwritten-digit images, 10 classes) — without hand-picking the
architecture.

### Approach

| Stage | What happens |
|-------|--------------|
| **Genome** (`src/ga_search.py`) | 1–3 hidden layers (sizes from {8,16,32,64,128}), activation (relu/tanh/logistic), L2 `alpha`, `learning_rate_init` |
| **Fitness** | validation accuracy of an MLP trained with that genome, minus a small penalty on total parameter count (so compactness breaks near-ties) |
| **Selection** | tournament (k=3) |
| **Crossover** | gene-wise uniform crossover — every child is a structurally valid genome, nothing to repair |
| **Mutation** | resize/add/remove a layer, swap activation, perturb `alpha`/learning rate — each gene independently, rate 0.3 |
| **Search budget** | population 16 × 12 generations, each candidate trained for 150 iterations on a held-out validation split (~50–75s total) |
| **Final step** | retrain the winning genome (and a fixed baseline) to convergence (400 iterations) on the full training set, evaluate both on a held-out test set |

### Results (held-out test set, n = 360)

| Model | Architecture | Parameters | Test accuracy |
|---|---|---:|:---:|
| Baseline MLP (fixed) | `(100,)` | 7,510 | **0.981** |
| **GA-evolved MLP** | `(32,)`, tanh | **2,410** | 0.972 |

The GA converged on a single 32-unit hidden layer with `tanh` activation —
**3.1× fewer parameters** than the hand-picked 100-unit baseline, within
**0.84 points** of its accuracy. It found this by generation 5 of 12 (see
convergence plot) and spent the rest of the search confirming nothing
smaller/better was reachable from there.

<p align="center">
  <img src="reports/figures/ga_convergence.png" width="48%">
  <img src="reports/figures/comparison_bar.png" width="46%">
</p>
<p align="center">
  <img src="reports/figures/confusion_matrix.png" width="45%">
</p>

## 2. Pure weight evolution — XOR, zero backprop (`src/neuroevolution_xor.py`)

A separate, much smaller demo: a fixed 2→4→1 network (17 weights total)
whose **weight vector is the genome**. There is no gradient descent
anywhere — fitness is classification accuracy on XOR, and evolution
(elitism + uniform crossover + Gaussian mutation) is the *only* optimizer.

```
gen   0  best_fitness=0.7379
gen  20  best_fitness=0.9962
gen  79  best_fitness=1.0000

final predictions on XOR:
  [0.0, 0.0] -> 0.008 (target 0)
  [0.0, 1.0] -> 0.998 (target 1)
  [1.0, 0.0] -> 0.985 (target 1)
  [1.0, 1.0] -> 0.013 (target 0)

accuracy: 100%  (17 weights evolved, zero gradient steps)
```

<p align="center"><img src="reports/figures/xor_convergence.png" width="55%"></p>

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/train.py                # runs the GA search + retrain + comparison (~1 min)
python src/neuroevolution_xor.py   # weight-evolution XOR bonus demo (~seconds)

streamlit run app/streamlit_app.py
```

## Deployment

The trained models (`models/*.joblib`) and report figures/metrics are
committed, so the Streamlit app runs with no training step. To deploy:

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io) → **New app** → point at
   this repo, main file `app/streamlit_app.py`.

## Project layout

```
src/ga_search.py            genome encoding + GA operators + fitness (architecture search)
src/train.py                 runs the search, retrains winner + baseline, writes models/figures
src/neuroevolution_xor.py    standalone weight-evolution demo (no backprop)
app/streamlit_app.py         interactive demo: try a digit, compare models, view the GA run
models/ reports/             trained models + metrics + figures (tracked)
```

## Notes & limitations

- Educational project on a small, clean dataset — a from-scratch GA is used
  deliberately (rather than a library like NEAT/DEAP) to keep the
  selection/crossover/mutation mechanics visible and easy to read.
- The GA searches a modest space (1–3 layers, 5 layer-size choices, 3
  activations) that runs in about a minute on a laptop; a larger search
  space would need a bigger population/generation budget or parallel
  fitness evaluation to stay fast.
- "Neuroevolution" here covers two different things on purpose: evolving
  *architecture* trained by backprop (`ga_search.py`), and evolving
  *weights* directly with no backprop (`neuroevolution_xor.py`) — the
  README calls out which is which throughout.
