import json
import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Kinyarwanda to English", page_icon="🌍", layout="wide")

API_KEY = os.environ.get("API_KEY", "kin-demo-2026")  # must match the key in the Colab server cell

EXAMPLES = [
    "ibi bivuze iki",
    "uwo mwana yahise asinzira",
    "ken yakomeje kuririmba iyo ndirimbo",
]

# ---------- optional result files ----------
results = {"before": [34.17, 16.27]}  # chrF, BLEU of base Gemma on the 100 held-out sentences
if os.path.exists("results.json"):
    with open("results.json") as f:
        results.update(json.load(f))
before, after = results.get("before"), results.get("after")

# ---------- sidebar: connection to the model server ----------
st.sidebar.header("Model server")
api_url = st.sidebar.text_input(
    "Server link (from the Colab cell)",
    value=os.environ.get("API_URL", ""),
    placeholder="https://xxxx.trycloudflare.com",
).strip().rstrip("/")

if st.sidebar.button("Check connection"):
    try:
        ok = requests.get(f"{api_url}/health", timeout=15).ok
        st.sidebar.success("Server is online" if ok else "Server answered with an error")
    except Exception:
        st.sidebar.error("Server is offline. Start the Colab server cell and paste the new link.")

st.sidebar.caption("The model runs on a GPU in Colab. This page sends your sentence to it.")

# ---------- header ----------
st.title("Kinyarwanda → English translator")
st.caption("Gemma 3 4B fine-tuned with LoRA")

# ---------- try the translator ----------
st.header("Try it")

if "text" not in st.session_state:
    st.session_state["text"] = ""

st.write("Examples:")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        st.session_state["text"] = ex

text = st.text_area("Kinyarwanda sentence", key="text", height=90)
compare = st.checkbox("Also show base Gemma (before fine-tuning)", value=True)

if st.button("Translate", type="primary"):
    if not text.strip():
        st.warning("Type a Kinyarwanda sentence first.")
    elif not api_url:
        st.error("Paste the server link in the sidebar first.")
    else:
        try:
            with st.spinner("Translating..."):
                r = requests.post(
                    f"{api_url}/translate",
                    json={"text": text.strip(), "compare": compare, "key": API_KEY},
                    timeout=180,
                )
            if r.status_code != 200:
                st.error(f"Server error: {r.text[:200]}")
            else:
                data = r.json()
                if compare and data.get("base") is not None:
                    c1, c2 = st.columns(2)
                    c1.subheader("Base Gemma (before)")
                    c1.info(data["base"])
                    c2.subheader("Fine-tuned (after)")
                    c2.success(data["finetuned"])
                else:
                    st.subheader("Fine-tuned translation")
                    st.success(data["finetuned"])
        except Exception:
            st.error("Could not reach the model server. Check that the Colab server cell is "
                     "running, then paste the current link in the sidebar.")

st.caption("Short, everyday sentences work best. Rare words and long sentences can be wrong.")

# ---------- results ----------
st.divider()
st.header("Results")
c1, c2 = st.columns(2)
if before:
    c1.metric("Before: chrF", before[0])
    c1.metric("Before: BLEU", before[1])
if after:
    c2.metric("After: chrF", after[0], delta=round(after[0] - before[0], 2))
    c2.metric("After: BLEU", after[1], delta=round(after[1] - before[1], 2))
else:
    c2.write("After-training scores will appear here once they are recorded.")

if os.path.exists("val_outputs.csv"):
    st.subheader("Held-out examples")
    ex = pd.read_csv("val_outputs.csv").rename(columns={
        "kin": "Kinyarwanda", "base": "Base Gemma", "finetuned": "Fine-tuned", "ref": "Reference"})
    n = st.slider("How many examples", 5, min(50, len(ex)), 10)
    st.dataframe(ex.head(n), use_container_width=True, hide_index=True)

st.divider()
st.header("Method and limits")
st.markdown(
    "- **Model:** Gemma 3 4B, LoRA r=16 on the attention and MLP layers (0.69% of parameters trained)\n"
    "- **Data:** about 2,850 cleaned pairs, 100 sentences held out for testing, loss on the answer only\n"
    "- **Works:** short everyday sentences, names kept, clean English output\n"
    "- **Fails:** rare words and long sentences, where the English is fluent but the meaning can be wrong\n"
    "- **Why:** a small dataset and a base model with little Kinyarwanda"
)
