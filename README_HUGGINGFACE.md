---
title: Yelp Ads Lab (CTR & auctions)
emoji: 🍽️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
python_version: "3.12"
pinned: false
short_description: DeepFM CTR, GSP auctions, bid optimizer Streamlit demo.
---

# Yelp Ads Lab

Streamlit UI for **CTR-style ranking**, a **GSP auction** simulator, **bid / budget** simulation, and a **model explorer** (calibration plot, embeddings, feature proxies).

**Models:** `models/*.pt` / `*.pkl` are gitignored, but **`scripts/push_hf_space.py` copies them from your machine** into the Space (`models/best_deepfm.pt`, `platt_scaler.pkl`, optional `deepfm.onnx`, `feature_engineer.pkl`). The app then **loads that checkpoint on startup** (cached in memory) and does **not** synthesize a new demo model. Use **`--require-trained-model`** to fail the deploy if `best_deepfm.pt` is missing. Use **`--no-model-weights`** for a smaller upload that relies on the runtime demo path.

## Deploy from your laptop (recommended)

1. **Token:** create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with **write** access.

2. **Login (pick one):**
   ```bash
   pip install "huggingface_hub>=0.23"
   huggingface-cli login
   ```
   Or: `export HF_TOKEN=hf_...`

3. **Space id** — use the **same name as your GitHub repo** (e.g. `yourname/yelp-ads-bidding-ctr` for repo `github.com/yourname/yelp-ads-bidding-ctr`). It must be unique under your user/org.

4. **First-time create + upload** from the **project root** (`yelp-ads-bidding-ctr/` that contains `app.py` and `src/`):
   ```bash
   cd /path/to/yelp-ads-bidding-ctr
   export HF_SPACE_REPO=yourname/yelp-ads-bidding-ctr
   .venv/bin/python3 scripts/push_hf_space.py --create
   ```
   Add `--private` if you want a private Space.

5. **Later updates** (same command without `--create` if the Space already exists). This **re-uploads your local trained weights** whenever they exist under `models/`:
   ```bash
   export HF_SPACE_REPO=yourname/yelp-ads-bidding-ctr
   .venv/bin/python3 scripts/push_hf_space.py --require-trained-model
   ```

The script stages `Dockerfile` (from `Dockerfile.huggingface`), `app.py`, `app/`, `src/`, `.streamlit/`, `requirements-huggingface.txt` as `requirements.txt`, and this file as `README.md`, then runs `upload_folder` to the Space.

**Why Docker?** Hugging Face’s **create Space** API no longer accepts `sdk: streamlit`; allowed values are `gradio`, `docker`, or `static`. This Space uses **Docker** and runs **Streamlit on port 7860** inside the container.

If `--create` failed earlier with a 400 error, delete any empty/broken Space on the Hub and run again, or create the Space once in the UI as **Docker** and then run the script **without** `--create`.

**Optional:** inspect the bundle before upload:
```bash
.venv/bin/python3 scripts/push_hf_space.py --repo-id yourname/yelp-ads-bidding-ctr --keep-staging ./build/hf_space_bundle
```

## Setup only on the Hugging Face website

1. Create a **Streamlit** Space (blank or from this repo).
2. Point the Space at this GitHub repository (or push these files).
3. Ensure **`requirements.txt`** matches **`requirements-huggingface.txt`** (CPU PyTorch) if the default build is too large or fails.
4. Confirm **`app.py`** is at the repository root with **`app/streamlit_app.py`** and **`src/`**.

## Local smoke test (not on HF)

```bash
cd /path/to/yelp-ads-bidding-ctr
PYTHONPATH=. .venv/bin/streamlit run app.py
```
