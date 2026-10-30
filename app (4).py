import json
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Kinyarwanda to English", page_icon="🌍", layout="wide")

# ---------- load results (files are optional) ----------
results = {"before": [34.17, 16.27]}  # chrF, BLEU of base Gemma on the 100 held-out sentences
if os.path.exists("results.json"):
    with open("results.json") as f:
        results.update(json.load(f))
before = results.get("before")
after = results.get("after")

# ---------- header ----------
st.title("Kinyarwanda → English translator")
st.caption("Gemma 3 4B fine-tuned with LoRA")

live_url = os.environ.get("LIVE_DEMO_URL")
if live_url:
    st.link_button("Open the live translator (runs on a GPU)", live_url, type="primary")
else:
    st.info("The live model runs on a GPU. This page shows the method, the scores and real examples.")

# ---------- project ----------
st.header("The project")
st.write(
    "A translator from Kinyarwanda to English, built by fine-tuning an open LLM "
    "(Gemma 3 4B) with LoRA on about 2,850 sentence pairs. Kinyarwanda is a low-resource "
    "language, so the base model knows little of it."
)

st.subheader("Method")
st.markdown(
    "- **Model:** Gemma 3 4B instruction-tuned\n"
    "- **Fine-tuning:** LoRA, r=16, on the attention and MLP layers (0.69% of parameters trained)\n"
    "- **Data:** cleaned pairs (stray commas and duplicates removed), 100 sentences held out for testing\n"
    "- **Loss:** computed on the English answer only, not on the prompt\n"
    "- **Metrics:** chrF and BLEU on the 100 held-out sentences, case-insensitive"
)

# ---------- results ----------
st.header("Results")
c1, c2, c3 = st.columns(3)
if before:
    c1.metric("Before: chrF", before[0])
    c1.metric("Before: BLEU", before[1])
if after:
    c2.metric("After: chrF", after[0], delta=round(after[0] - before[0], 2))
    c2.metric("After: BLEU", after[1], delta=round(after[1] - before[1], 2))
else:
    c2.write("After-training scores will appear here once they are recorded.")

# ---------- examples ----------
st.header("Examples")
if os.path.exists("val_outputs.csv"):
    ex = pd.read_csv("val_outputs.csv")
    ex = ex.rename(columns={"kin": "Kinyarwanda", "base": "Base Gemma",
                            "finetuned": "Fine-tuned", "ref": "Reference"})
    n = st.slider("How many examples", 5, min(50, len(ex)), 10)
    st.dataframe(ex.head(n), use_container_width=True, hide_index=True)
else:
    st.caption("Early test run (30 training steps), held-out sentences:")
    ex = pd.DataFrame(
        [
            ["ken yakomeje kuririmba iyo ndirimbo", "Keep singing that song.",
             "ken continued singing that song", "ken continued to sing that song"],
            ["koga ntibyemewe muri uru ruzi", "I'm not fluent in that language.",
             "don't do that in this class", "swimming is not allowed in this river"],
            ["hano hari umwanya uhagije wo kubika.", "It's a good time to write.",
             "there is a good place to write here.", "there is enough storage space here."],
        ],
        columns=["Kinyarwanda", "Base Gemma", "Fine-tuned", "Reference"],
    )
    st.dataframe(ex, use_container_width=True, hide_index=True)

# ---------- limits ----------
st.header("What works and what does not")
st.markdown(
    "- **Works:** short everyday sentences, names kept, clean English output, no extra text.\n"
    "- **Fails:** rare words and long sentences. The English is fluent but the meaning can be wrong.\n"
    "- **Why:** a small dataset (about 2,850 pairs) and a base model with little Kinyarwanda."
)

st.header("Next steps")
st.markdown(
    "- More and cleaner Kinyarwanda-English data\n"
    "- Test a stronger base model (Gemma 3 12B, TranslateGemma)\n"
    "- Compare with a dedicated translation model such as NLLB-200"
)
