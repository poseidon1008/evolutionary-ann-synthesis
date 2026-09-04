"""Three demos in one app:
  1. Try a digit — GA-evolved architecture (via backprop) vs. a fixed baseline
  2. Compare 4 algorithms — GA vs ES vs PSO vs NEAT-lite, same task, no backprop
  3. XOR bonus — pure weight-evolution, zero gradient steps

    streamlit run app/streamlit_app.py
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"

st.set_page_config(page_title="Evolutionary ANN Synthesis", page_icon="🧬", layout="centered")


@st.cache_resource
def load_artifacts():
    evolved = joblib.load(MODEL_DIR / "evolved_mlp.joblib")
    baseline = joblib.load(MODEL_DIR / "baseline_mlp.joblib")
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
    return evolved, baseline, metrics


@st.cache_data
def load_comparison():
    path = ROOT / "reports" / "evo_comparison.json"
    return json.loads(path.read_text()) if path.exists() else None


st.title("🧬 Evolutionary ANN Synthesis")
st.caption(
    "Building neural networks with population-based, gradient-free search "
    "instead of (or alongside) backpropagation."
)

try:
    evolved, baseline, metrics = load_artifacts()
except FileNotFoundError:
    st.error("No trained models found — run `python src/train.py` first, then reload.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Try a digit", "Compare 4 algorithms", "XOR bonus"])

# ---------------------------------------------------------------------------
# Tab 1 — GA architecture search (trained via backprop) vs. fixed baseline
# ---------------------------------------------------------------------------
with tab1:
    st.write(
        "A genetic algorithm searched MLP architectures/hyperparameters; the "
        "winner (and a fixed baseline) were then trained with ordinary "
        "backprop. This tab is about the GA finding a good **design** — "
        "training itself is still gradient descent."
    )

    digits = load_digits()
    X, y = digits.data, digits.target
    X_trainfull, X_test, y_trainfull, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    idx_all = np.arange(len(X))
    _, idx_test = train_test_split(idx_all, test_size=0.2, stratify=y, random_state=42)
    test_images = digits.images[idx_test]

    st.subheader("Try a digit")
    i = st.slider("Test-set digit index", 0, len(X_test) - 1, 0)
    col1, col2 = st.columns([1, 2])
    with col1:
        fig, ax = plt.subplots(figsize=(2.2, 2.2))
        ax.imshow(test_images[i], cmap="gray_r")
        ax.axis("off")
        st.pyplot(fig)
        st.caption(f"True label: **{y_test[i]}**")

    with col2:
        x = X_test[i:i + 1]
        ev_scaled = evolved["scaler"].transform(x)
        bl_scaled = baseline["scaler"].transform(x)
        ev_pred = evolved["model"].predict(ev_scaled)[0]
        bl_pred = baseline["model"].predict(bl_scaled)[0]
        ev_conf = evolved["model"].predict_proba(ev_scaled)[0][ev_pred]
        bl_conf = baseline["model"].predict_proba(bl_scaled)[0][bl_pred]

        st.metric(f"Evolved MLP {tuple(metrics['evolved']['genome']['hidden_layers'])}",
                  f"predicts {ev_pred}", f"{ev_conf:.0%} confidence")
        st.metric("Baseline MLP (100,)",
                  f"predicts {bl_pred}", f"{bl_conf:.0%} confidence")

    st.divider()
    st.subheader("Evolved vs. baseline")
    c1, c2, c3 = st.columns(3)
    c1.metric("Evolved test accuracy", f"{metrics['evolved']['test_accuracy']:.1%}")
    c2.metric("Baseline test accuracy", f"{metrics['baseline']['test_accuracy']:.1%}")
    c3.metric("Params: evolved vs baseline",
              f"{metrics['evolved']['params']:,}",
              f"{metrics['evolved']['params'] - metrics['baseline']['params']:,} vs baseline")

    st.image(str(FIG_DIR / "comparison_bar.png"))
    st.image(str(FIG_DIR / "ga_convergence.png"))

    with st.expander("Winning genome found by the GA"):
        st.json(metrics["evolved"]["genome"])
        st.write(f"Search: {metrics['search']['pop_size']} genomes × "
                 f"{metrics['search']['generations']} generations "
                 f"in {metrics['search']['seconds']}s.")

# ---------------------------------------------------------------------------
# Tab 2 — GA vs ES vs PSO vs NEAT-lite, no backprop anywhere
# ---------------------------------------------------------------------------
with tab2:
    st.write(
        "Four population-based, **gradient-free** algorithms synthesizing the "
        "same digit classifier — no backprop anywhere on this tab. GA, ES and "
        "PSO all optimize a fixed `(64, 24, 10)` weight vector; NEAT-lite "
        "starts with no hidden nodes at all and grows its own topology."
    )
    comparison = load_comparison()
    if comparison is None:
        st.info("Run `python src/compare_evolution.py` to generate this comparison.")
    else:
        cfg = comparison["config"]
        res = comparison["results"]
        st.caption(f"Population {cfg['pop_size']} × {cfg['generations']} generations, "
                    f"identical for all four algorithms.")

        cols = st.columns(4)
        for col, name in zip(cols, ["GA", "ES", "PSO", "NEAT-lite"]):
            r = res[name]
            extra = f"{r['hidden_nodes']} hidden, {r['connections']} conn" if name == "NEAT-lite" else f"{r['params']:,} params"
            col.metric(name, f"{r['test_accuracy']:.1%}", extra)

        st.image(str(FIG_DIR / "evo_compare_convergence.png"))
        c1, c2 = st.columns(2)
        c1.image(str(FIG_DIR / "evo_compare_accuracy.png"))
        c2.image(str(FIG_DIR / "evo_compare_size.png"))

        with st.expander("What's actually different about each algorithm"):
            st.markdown("""
- **GA** — tournament selection, real-coded blend (BLX-α) crossover between
  two parents, Gaussian mutation. The classic "population + crossover +
  mutation" recipe.
- **ES** — `(μ/μ, λ)` evolution strategy: no crossover between individuals
  at all; instead the next generation's mean is a weighted average of the
  fittest `μ` offspring, and the mutation strength (`σ`) self-adapts via a
  log-normal draw each generation.
- **PSO** — not evolutionary in the Darwinian sense (no selection, no
  offspring — every particle survives forever). Each particle just moves
  toward a blend of its own best-ever position and the swarm's best.
- **NEAT-lite** — the only one searching network *shape*, not just weights
  for a shape you chose. Genomes start as a plain input→output linear
  model and grow hidden nodes/connections via structural mutation —
  mutation-only (no crossover, which in full NEAT needs innovation-number
  bookkeeping to match up genomes; out of scope here).
""")
        st.caption(
            "Ranking (ES > GA > PSO > NEAT-lite on accuracy) matches the "
            "textbook expectation for a smooth, ~1,800-dimensional continuous "
            "landscape: ES's self-adaptive step size and weighted "
            "recombination are purpose-built for exactly this. NEAT-lite "
            "trades accuracy for a network **~6× smaller** — and was still "
            "climbing when the shared generation budget ran out, since "
            "growing structure from scratch is a harder search than tuning "
            "weights for a size you were handed."
        )

# ---------------------------------------------------------------------------
# Tab 3 — pure weight-evolution on XOR
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Pure weight-evolution on XOR (no backprop)")
    st.write(
        "A separate, smaller demo — a fixed 2→4→1 network whose **weights** "
        "are the genome. No gradient descent anywhere; selection + "
        "crossover + mutation alone find weights that solve XOR. Run "
        "`python src/neuroevolution_xor.py` to regenerate."
    )
    xor_fig = FIG_DIR / "xor_convergence.png"
    if xor_fig.exists():
        st.image(str(xor_fig))
    else:
        st.info("Run `python src/neuroevolution_xor.py` to generate this figure.")
