"""Multi-page Streamlit app: dashboard, auction simulator, bid optimizer, model explorer."""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Project root (parent of app/)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.auction.bid_optimizer import CampaignSimulator  # noqa: E402
from src.inference.demo_assets import write_demo_artifacts  # noqa: E402
from src.inference.pipeline import default_model_paths  # noqa: E402
from src.models.calibration import PlattScaler, compute_ece  # noqa: E402

# --- Demo catalog (indices align with demo model vocab 12 x 8) ---
CUISINE_LABELS = [
    "Japanese",
    "Italian",
    "Mexican",
    "Thai",
    "American",
    "Indian",
    "French",
    "Korean",
    "Vietnamese",
    "Mediterranean",
    "BBQ",
    "Seafood",
]
CITY_LABELS = ["Phoenix", "Austin", "Chicago", "Denver", "Seattle", "Miami", "Boston", "Portland"]
TIME_SLOTS = {
    "Morning": "2025-06-15 09:30:00",
    "Lunch": "2025-06-15 12:30:00",
    "Evening": "2025-06-15 18:45:00",
    "Night": "2025-06-15 21:15:00",
}


@st.cache_resource
def load_inference_bundle():
    """Load production checkpoint if possible; otherwise materialize demo assets once."""
    from src.inference.pipeline import InferencePipeline

    paths = default_model_paths()
    model_p = paths["model"]
    eng_p = paths["engineer"]
    platt_p = paths["platt"]
    onnx_p = paths["onnx"]

    if model_p.is_file():
        try:
            pipe = InferencePipeline(
                model_path=model_p,
                engineer_path=eng_p if eng_p.is_file() else None,
                platt_path=platt_p if platt_p.is_file() else None,
                onnx_path=onnx_p if onnx_p.is_file() else None,
                use_onnx=onnx_p.is_file(),
            )
            return pipe, "Loaded production checkpoint from `models/`."
        except Exception as exc:
            st.session_state["_load_warn"] = str(exc)

    tmp = Path(tempfile.mkdtemp(prefix="streamlit_yelp_demo_"))
    try:
        m, p, o = write_demo_artifacts(tmp)
        pipe = InferencePipeline(
            model_path=m,
            engineer_path=None,
            platt_path=p,
            onnx_path=o,
            use_onnx=True,
        )
        msg = (
            "No compatible production checkpoint found — using **demo** DeepFM + ONNX "
            f"(fast path). Last error: `{st.session_state.pop('_load_warn', 'none')}`."
        )
        return pipe, msg
    except Exception as exc:
        return None, f"Could not load any model: `{exc}`"


def _dashboard_data() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    kpis = {
        "impressions": 1_248_500,
        "clicks": 89_340,
        "revenue": 412_880.50,
        "cpc": 4.62,
    }
    ctr_by_cuisine = pd.DataFrame(
        {
            "cuisine": CUISINE_LABELS[:8],
            "ctr": [0.042, 0.055, 0.038, 0.061, 0.033, 0.049, 0.052, 0.041],
        }
    )
    revenue_strategy = pd.DataFrame(
        {
            "strategy": ["Random", "Pure Bid", "Pure CTR", "Quality (bid×CTR)"],
            "revenue_k": [247.3, 289.1, 265.4, 318.2],
        }
    )
    top_restaurants = pd.DataFrame(
        {
            "restaurant": [f"r_{c.lower()}_prime" for c in CUISINE_LABELS[:6]],
            "clicks": [12400, 10200, 9800, 9100, 8600, 7900],
        }
    ).sort_values("clicks", ascending=True)
    return kpis, ctr_by_cuisine, revenue_strategy, top_restaurants


def page_dashboard() -> None:
    st.header("Dashboard")
    kpis, ctr_c, rev_s, top_r = _dashboard_data()
    ctr_rate = kpis["clicks"] / max(kpis["impressions"], 1)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Impressions", f"{kpis['impressions']:,}")
    c2.metric("Clicks", f"{kpis['clicks']:,}")
    c3.metric("CTR", f"{ctr_rate:.2%}")
    c4.metric("Revenue", f"${kpis['revenue']:,.0f}")
    c5.metric("Avg CPC", f"${kpis['cpc']:.2f}")

    col_a, col_b = st.columns(2)
    with col_a:
        fig1 = go.Figure(
            data=go.Bar(x=ctr_c["cuisine"], y=ctr_c["ctr"], marker_color="#3498db")
        )
        fig1.update_layout(
            title="CTR by cuisine",
            yaxis_tickformat=".1%",
            height=400,
            margin=dict(t=50, b=80),
        )
        st.plotly_chart(fig1, use_container_width=True)
    with col_b:
        fig2 = go.Figure(
            data=go.Bar(
                x=rev_s["strategy"],
                y=rev_s["revenue_k"],
                marker_color=["#95a5a6", "#3498db", "#2ecc71", "#e74c3c"],
            )
        )
        fig2.update_layout(
            title="Revenue by ranking strategy (sample, $K)",
            height=400,
            margin=dict(t=50, b=80),
        )
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = go.Figure(
        data=go.Bar(
            x=top_r["clicks"],
            y=top_r["restaurant"],
            orientation="h",
            marker_color="#9b59b6",
        )
    )
    fig3.update_layout(
        title="Top restaurants by clicks",
        height=380,
        margin=dict(l=120, t=50),
    )
    st.plotly_chart(fig3, use_container_width=True)


def _auction_candidates(
    cuisine_pref: str,
    city_name: str,
) -> list[dict]:
    pref_i = CUISINE_LABELS.index(cuisine_pref) if cuisine_pref in CUISINE_LABELS else 0
    city_i = CITY_LABELS.index(city_name) if city_name in CITY_LABELS else 0
    base: list[dict] = []
    bids = [4.2, 3.1, 2.8, 2.2, 3.5, 1.9, 2.6, 3.8]
    for k, cu in enumerate([0, 1, 2, 3, 4, 5, 6, 7]):
        ci = (city_i + k) % len(CITY_LABELS)
        rating = 0.45 + 0.08 * (1 if cu == pref_i else 0) + 0.02 * (k % 3)
        base.append(
            {
                "restaurant_id": f"r_{CUISINE_LABELS[cu].lower()}_{k}",
                "business_id": f"b_{k}",
                "campaign_id": f"c_{k}",
                "bid": bids[k],
                "cuisine": cu,
                "city": ci,
                "norm_rating": min(rating, 0.95),
            }
        )
    return base


def page_auction() -> None:
    st.header("Auction Simulator")
    pipe, msg = load_inference_bundle()
    st.caption(msg)

    with st.sidebar:
        st.subheader("User profile")
        cuisine_pref = st.selectbox("Cuisine preference", CUISINE_LABELS, index=0)
        city = st.selectbox("City", CITY_LABELS, index=0)
        tod = st.selectbox("Time of day", list(TIME_SLOTS.keys()), index=2)
        run = st.button("Run auction", type="primary")

    if pipe is None:
        st.error(
            "Inference pipeline is unavailable. Add a checkpoint under `models/` or check logs."
        )
        return

    if run:
        ts = TIME_SLOTS[tod]
        candidates = _auction_candidates(cuisine_pref, city)
        t0 = time.perf_counter()
        try:
            rows = pipe.get_sponsored_listings(
                {"user_id": "streamlit_user"},
                {"timestamp": ts},
                candidates,
            )
        except Exception as exc:
            st.error(f"Inference failed: `{exc}`")
            return
        elapsed = time.perf_counter() - t0
        st.success(f"Completed in **{elapsed:.2f}s**.")
        if elapsed < 2.0:
            st.caption("Within the 2s target for the demo ONNX inference path.")

        df = pd.DataFrame(rows)
        if df.empty:
            st.warning("No sponsored listings returned (check CTR predictions).")
            return
        st.dataframe(
            df,
            column_config={
                "position": st.column_config.NumberColumn("Rank", format="%d"),
                "restaurant": st.column_config.TextColumn("Restaurant"),
                "predicted_ctr": st.column_config.NumberColumn("Pred. CTR", format="%.5f"),
                "price": st.column_config.NumberColumn("GSP price", format="$%.4f"),
            },
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Set profile options in the sidebar and click **Run auction**.")


def page_bid_optimizer() -> None:
    st.header("Bid Optimizer")
    c1, c2, c3 = st.columns(3)
    with c1:
        budget = st.number_input("Daily budget ($)", min_value=10.0, value=150.0, step=10.0)
    with c2:
        cpa = st.number_input("CPA target / value per click ($)", min_value=0.5, value=4.0, step=0.5)
    with c3:
        duration = st.slider("Duration (rounds)", min_value=6, max_value=48, value=24)

    if st.button("Simulate campaign", type="primary"):
        sim = CampaignSimulator(
            daily_budget=budget,
            value_per_click=cpa,
            n_rounds=duration,
            impressions_per_round=40,
            seed=42,
        )
        traj = sim.simulate()
        final = traj.iloc[-1]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Budget utilization", f"{final['budget_utilization']*100:.1f}%")
        m2.metric("Total spend", f"${final['total_spend']:.2f}")
        m3.metric("Clicks", f"{int(final['total_clicks'])}")
        m4.metric("Actual CPA", f"${final['actual_CPA']:.2f}")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=traj["round"],
                y=traj["budget_utilization"] * 100,
                mode="lines",
                name="Utilization %",
                line=dict(color="#e74c3c", width=2),
            )
        )
        fig.update_layout(
            title="Budget utilization over rounds",
            xaxis_title="Round",
            yaxis_title="Utilization (%)",
            height=400,
            yaxis_range=[0, 105],
        )
        st.plotly_chart(fig, use_container_width=True)


def _calibration_figure() -> go.Figure:
    rng = np.random.default_rng(42)
    n = 6000
    y_true = rng.binomial(1, 0.12, n).astype(float)
    logits = rng.normal(-2.0, 1.2, n) + y_true * 0.8
    raw = 1.0 / (1.0 + np.exp(-logits))
    raw = np.clip(raw, 1e-6, 1.0 - 1e-6)

    platt = PlattScaler()
    platt.fit(y_true, raw)
    cal = platt.calibrate(raw)

    br = compute_ece(y_true, raw, n_bins=10)
    ac = compute_ece(y_true, cal, n_bins=10)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Before Platt (ECE ≈ {br.ece:.3f})",
            f"After Platt (ECE ≈ {ac.ece:.3f})",
        ),
    )
    for col, rel, title_suf in (
        (1, br, "raw"),
        (2, ac, "calibrated"),
    ):
        mask = rel.bin_count > 0
        xf = rel.bin_mean_pred[mask]
        yf = rel.bin_mean_true[mask]
        fig.add_trace(
            go.Scatter(
                x=xf,
                y=yf,
                mode="markers+lines",
                name=f"Empirical ({title_suf})",
                marker=dict(size=10),
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="gray"),
                name="Perfect",
                showlegend=(col == 1),
            ),
            row=1,
            col=col,
        )
    fig.update_xaxes(title_text="Mean predicted", range=[0, 1])
    fig.update_yaxes(title_text="Mean observed", range=[0, 1])
    fig.update_layout(height=420, margin=dict(t=60))
    return fig


def _embedding_tsne_figure(pipe) -> go.Figure | None:
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        return None

    if pipe is None or pipe.model is None:
        return None

    emb_layer = pipe.model.embedding_layer
    # Prefer cuisine-like feature
    key = None
    for name in ("cuisine", "restaurant_cuisine", "primary_cuisine"):
        if name in emb_layer.embeddings:
            key = name
            break
    if key is None:
        key = next(iter(emb_layer.embeddings.keys()))

    mat = emb_layer.embeddings[key].weight.detach().cpu().numpy()
    n = min(200, mat.shape[0])
    mat = mat[:n]
    labels = [f"{key}_{i}" for i in range(n)]
    perplexity = max(5, min(30, n - 1))
    z = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(mat)
    fig = go.Figure(
        data=go.Scatter(
            x=z[:, 0],
            y=z[:, 1],
            mode="markers",
            marker=dict(size=7, color=np.arange(n), colorscale="Viridis", showscale=True),
            text=labels,
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"t-SNE of `{key}` embeddings (n={n})",
        height=450,
    )
    return fig


def _feature_importance_figure(pipe) -> go.Figure | None:
    if pipe is None or pipe.model is None:
        return None
    names: list[str] = []
    scores: list[float] = []

    for fname, mod in pipe.model.embedding_layer.embeddings.items():
        w = mod.weight.detach().cpu().numpy()
        names.append(f"sparse:{fname}")
        scores.append(float(np.linalg.norm(w)))

    dl = pipe.model.dense_linear.weight.detach().cpu().numpy().ravel()
    dense_feats = pipe.model.dense_features
    for i, fn in enumerate(dense_feats):
        names.append(f"dense:{fn}")
        scores.append(float(abs(dl[i])))

    df = pd.DataFrame({"feature": names, "score": scores}).sort_values("score", ascending=True)
    fig = go.Figure(
        go.Bar(x=df["score"], y=df["feature"], orientation="h", marker_color="#16a085")
    )
    fig.update_layout(title="Feature signal strength (embedding L2 / dense |weight|)", height=500)
    return fig


def page_explorer() -> None:
    st.header("Model Explorer")
    pipe, msg = load_inference_bundle()
    st.caption(msg)

    st.subheader("Calibration (reliability)")
    st.plotly_chart(_calibration_figure(), use_container_width=True)
    st.caption(
        "Synthetic validation sample: **lower ECE after Platt** indicates better alignment "
        "between predicted probability and observed click rate."
    )

    st.subheader("Restaurant / cuisine embeddings (t-SNE)")
    tsne_fig = _embedding_tsne_figure(pipe)
    if tsne_fig is not None:
        st.plotly_chart(tsne_fig, use_container_width=True)
    else:
        st.warning("t-SNE unavailable (no model or sklearn missing).")

    st.subheader("Feature importance (proxy)")
    fi_fig = _feature_importance_figure(pipe)
    if fi_fig is not None:
        st.plotly_chart(fi_fig, use_container_width=True)
    else:
        st.warning("Feature importance unavailable without a loaded model.")


def main() -> None:
    st.set_page_config(
        page_title="Yelp Ads Lab",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Page",
        ["Dashboard", "Auction Simulator", "Bid Optimizer", "Model Explorer"],
        label_visibility="collapsed",
    )

    if page == "Dashboard":
        page_dashboard()
    elif page == "Auction Simulator":
        page_auction()
    elif page == "Bid Optimizer":
        page_bid_optimizer()
    else:
        page_explorer()


if __name__ == "__main__":
    main()
