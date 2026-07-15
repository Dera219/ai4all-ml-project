"""Streamlit deployment for the Nutrition5k calorie classifier.

Run:  streamlit run app/streamlit_app.py

Week 10 deliverable. Loads the packaged checkpoint from 'Seeded Model.ipynb' and classifies an
uploaded food photo as Low / Medium / High calorie.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.ordinal_cnn import CLASS_NAMES, CalorieClassifier  # noqa: E402

CHECKPOINT = REPO_ROOT / "models" / "final_ordinal_multitask_model.pth"
SCALER = REPO_ROOT / "models" / "regression_target_scaler.pkl"

st.set_page_config(page_title="Food Calorie Classifier", page_icon="🍽️", layout="centered")


@st.cache_resource
def load_model() -> CalorieClassifier | None:
    """Cached so the checkpoint is read once, not on every interaction."""
    try:
        return CalorieClassifier.load(CHECKPOINT, SCALER)
    except (FileNotFoundError, ValueError):
        return None


def render_missing_weights() -> None:
    st.error("Model weights not found — the app cannot make predictions.")
    st.markdown(
        f"""
The trained weights are **not in this repository**. They were written to `/kaggle/working/`
during training, which Kaggle wipes when the session ends, and were never committed.

**To fix:** open `Seeded Model.ipynb` on Kaggle, run it, and download from the output panel:

- `final_ordinal_multitask_model.pth`
- `regression_target_scaler.pkl`

Place both in `{CHECKPOINT.parent.relative_to(REPO_ROOT)}/`.

See `EXPORT_WEIGHTS.md` for the full procedure.
"""
    )


def render_honest_caveats(metadata: dict) -> None:
    test_acc = metadata.get("test_accuracy")
    with st.expander("How much should you trust this? (read before demoing)", expanded=False):
        st.markdown(
            f"""
**Reported test accuracy: {test_acc:.1%}** against a 33.3% random baseline on three balanced
classes. That is a real result — roughly 2.2x baseline.

**But treat it as optimistic, not as a clean held-out estimate.** The test set was evaluated
against repeatedly while the architecture was being iterated (accuracies of 66.9%, 70.0%, 71.3%,
71.5%, 72.3% appear across the notebooks before this model's 74.1%). Once you have looked at the
test set several times and kept the change that improved it, you have selected on it, and it is
no longer measuring generalization. The honest framing is *"we improved against the test set
across iterations, so the true out-of-sample number is likely lower."*

**What was verified clean:** `dish-level-data-leakage-check.ipynb` confirms 0 `dish_id` overlap
between train/val/test. Since each dish has exactly one overhead RGB image, the split is sound at
the dish level.

**Where it will be least accurate:** Nutrition5k was captured in a controlled setting with a
specific culinary range and a fixed overhead camera. Photos from a phone at an angle, in
different lighting, of cuisines the dataset under-represents — all fall outside what the model
saw. Food datasets skew Western by default, so the people this serves worst are those whose food
it never saw.

**This is a class project, not a dietary tool.** Do not make food decisions from it.
"""
        )


def main() -> None:
    st.title("🍽️ Food Calorie Classifier")
    st.caption("Nutrition5k · ordinal multi-task CNN · AI4ALL")

    classifier = load_model()
    if classifier is None:
        render_missing_weights()
        st.stop()

    render_honest_caveats(classifier.metadata)

    uploaded = st.file_uploader(
        "Upload a food photo",
        type=["png", "jpg", "jpeg"],
        help="Works best on an overhead shot of a single plate — that is what it was trained on.",
    )

    if uploaded is None:
        st.info("Upload an image to classify it as Low, Medium, or High calorie.")
        return

    image = Image.open(uploaded)
    st.image(image, caption="Input", use_container_width=True)

    with st.spinner("Classifying..."):
        prediction = classifier.predict(image)

    st.subheader(f"Prediction: {prediction.class_name} calorie")

    if not prediction.is_consistent:
        # [0, 1] — "not above threshold 1, but above threshold 2" is incoherent for ordered
        # classes. Surfacing it is more honest than printing a confident label.
        st.warning(
            "The two ordinal thresholds disagree with each other, so this prediction is "
            "internally inconsistent. Treat it as the model being confused rather than as a "
            "real answer."
        )

    p1, p2 = prediction.threshold_probabilities
    st.markdown("**Threshold probabilities**")
    st.markdown(
        f"- P(above Low) = `{p1:.3f}` {'✅' if p1 >= 0.5 else '❌'}\n"
        f"- P(above Medium) = `{p2:.3f}` {'✅' if p2 >= 0.5 else '❌'}"
    )
    st.caption(
        f"Class = number of thresholds passed. {CLASS_NAMES[0]}=[0,0], "
        f"{CLASS_NAMES[1]}=[1,0], {CLASS_NAMES[2]}=[1,1]."
    )

    if prediction.nutrition is not None:
        st.markdown("**Auxiliary regression estimates**")
        cols = st.columns(len(prediction.nutrition))
        units = {"calories": "kcal", "mass": "g", "fat": "g", "carb": "g", "protein": "g"}
        for col, (name, value) in zip(cols, prediction.nutrition.items(), strict=True):
            col.metric(name.capitalize(), f"{value:.0f} {units.get(name, '')}")
        st.caption(
            "These come from the auxiliary head, which exists to give the shared trunk a richer "
            "training signal. They were never tuned for accuracy — read them as a rough sanity "
            "check, not as nutrition facts."
        )
    else:
        st.info(
            "Scaler not loaded, so the regression head's output is in standardized units and "
            "would be meaningless as nutrition numbers. Add `regression_target_scaler.pkl` to "
            "`models/` to see them."
        )


if __name__ == "__main__":
    main()
