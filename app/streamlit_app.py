"""Interactive demo: pick a handwritten digit, compare the GA-evolved
compact MLP against the fixed baseline MLP, and see the GA's search
convergence and the winning genome.

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


st.title("🧬 Evolutionary ANN Synthesis")
st.caption(
    "A genetic algorithm searches neural-network architectures/hyperparameters "
    "(the digits classifier below); a second GA evolves raw network *weights* "
    "with zero backprop (the XOR demo at the bottom)."
)

try:
    evolved, baseline, metrics = load_artifacts()
except FileNotFoundError:
    st.error("No trained models found — run `python src/train.py` first, then reload.")
    st.stop()

digits = load_digits()
X, y = digits.data, digits.target
X_trainfull, X_test, y_trainfull, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Recover the matching test-set *images* (same split/seed as src/train.py).
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

st.divider()
st.subheader("Bonus: pure weight-evolution on XOR (no backprop)")
st.write(
    "A separate, smaller demo — a fixed 2→4→1 network whose **weights** are "
    "the genome. No gradient descent anywhere; selection + crossover + "
    "mutation alone find weights that solve XOR. Run "
    "`python src/neuroevolution_xor.py` to regenerate."
)
xor_fig = FIG_DIR / "xor_convergence.png"
if xor_fig.exists():
    st.image(str(xor_fig))
else:
    st.info("Run `python src/neuroevolution_xor.py` to generate this figure.")
