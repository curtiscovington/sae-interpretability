from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path("outputs/topk_sweep")


def available_runs() -> list[tuple[str, str]]:
    runs: list[tuple[str, str]] = []
    if not ROOT.exists():
        return runs
    for k_dir in sorted(ROOT.glob("k*")):
        for layer_dir in sorted(k_dir.glob("layer_*")):
            if (layer_dir / "tables" / "feature_map_points.csv").exists():
                runs.append((k_dir.name, layer_dir.name))
    return runs


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _snip(x: str, n: int = 120) -> str:
    return (x or "").replace("\n", " ")[:n]


def add_annotations(df: pd.DataFrame, fa: dict, fb: dict) -> pd.DataFrame:
    out = df.copy()
    out["feature"] = out["feature"].astype(int)

    la, lb, ca, cb = [], [], [], []
    for fid in out["feature"]:
        a = fa.get(str(fid), {})
        b = fb.get(str(fid), {})
        la.append(a.get("heuristic_label", ""))
        lb.append(b.get("heuristic_label", ""))
        ca.append((a.get("top_contexts", [""]) or [""])[0])
        cb.append((b.get("top_contexts", [""]) or [""])[0])

    out["label_a"] = la
    out["label_b"] = lb
    out["ctx_a"] = ca
    out["ctx_b"] = cb
    out["abs_sel"] = out["selectivity_mag_AminusB"].abs()

    def concept(row):
        if row.selectivity_mag_AminusB > 0.02:
            return f"A-leaning · {row.label_a or row.label_b or 'unlabeled'}"
        if row.selectivity_mag_AminusB < -0.02:
            return f"B-leaning · {row.label_b or row.label_a or 'unlabeled'}"
        return f"Shared · {row.label_a or row.label_b or 'unlabeled'}"

    out["concept"] = out.apply(concept, axis=1)
    out["snippet"] = [
        _snip(a if (s > 0.02) else b if (s < -0.02) else (a or b))
        for a, b, s in zip(out["ctx_a"], out["ctx_b"], out["selectivity_mag_AminusB"])
    ]
    return out


def load_run(k: str, layer: str):
    base = ROOT / k / layer
    df = pd.read_csv(base / "tables" / "feature_map_points.csv")
    csum = pd.read_csv(base / "tables" / "feature_cluster_summary.csv")
    fa = load_json(base / "features" / "features_A.json")
    fb = load_json(base / "features" / "features_B.json")
    res = load_json(base / "results.json")
    return base, add_annotations(df, fa, fb), csum, fa, fb, res


def nearest(df: pd.DataFrame, fid: int, n: int = 10) -> pd.DataFrame:
    row = df[df.feature == fid]
    if row.empty:
        return pd.DataFrame()
    x, y, z = row.iloc[0][["pc1", "pc2", "pc3"]]
    d = np.sqrt((df.pc1 - x) ** 2 + (df.pc2 - y) ** 2 + (df.pc3 - z) ** 2)
    out = df.assign(distance=d).sort_values("distance").head(n + 1)
    return out[out.feature != fid].head(n)


def cluster_labels(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c, g in df.groupby("cluster"):
        top = g.sort_values("abs_sel", ascending=False).head(8)
        sample_terms = []
        for x in top["concept"].tolist():
            right = x.split("·", 1)[-1].strip()
            if right and right != "unlabeled":
                sample_terms.append(right)
        label = " | ".join(sample_terms[:2]) if sample_terms else "mixed"
        rows.append(
            {
                "cluster": int(c),
                "n": int(len(g)),
                "mean_sel": float(g.selectivity_mag_AminusB.mean()),
                "label_guess": label,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_sel", ascending=False)


st.set_page_config(page_title="SAE Interpretability Explorer", layout="wide")
st.title("SAE Interpretability Explorer")
st.caption("Designed for concept discovery: map → filter → inspect → neighborhood.")

if "selected_feature" not in st.session_state:
    st.session_state.selected_feature = None

runs = available_runs()
if not runs:
    st.error("No runs found. Generate feature map tables first.")
    st.stop()

options = [f"{k}/{l}" for k, l in runs]
run = st.sidebar.selectbox("Run", options, index=max(len(options) - 1, 0))
k, layer = run.split("/")
base, df, csum, fa, fb, res = load_run(k, layer)

st.sidebar.markdown("### Global Filters")
sel_type = st.sidebar.selectbox("Selectivity type", ["all", "A-leaning", "B-leaning", "shared"])
sel_thr = st.sidebar.slider("selectivity threshold", 0.0, 2.0, 0.05, 0.01)
search = st.sidebar.text_input("Search labels/snippets", "").strip().lower()

view = df.copy()
if sel_type == "A-leaning":
    view = view[view.selectivity_mag_AminusB >= sel_thr]
elif sel_type == "B-leaning":
    view = view[view.selectivity_mag_AminusB <= -sel_thr]
elif sel_type == "shared":
    view = view[view.selectivity_mag_AminusB.abs() <= sel_thr]
if search:
    view = view[
        view["concept"].str.lower().str.contains(search, na=False)
        | view["label_a"].str.lower().str.contains(search, na=False)
        | view["label_b"].str.lower().str.contains(search, na=False)
        | view["snippet"].str.lower().str.contains(search, na=False)
    ]

m1, m2, m3, m4 = st.columns(4)
if res:
    m1.metric("A→A R²", f"{res.get('trainA_evalA', {}).get('r2', float('nan')):.3f}")
    m2.metric("B→B R²", f"{res.get('trainB_evalB', {}).get('r2', float('nan')):.3f}")
    m3.metric("A→B MSE ratio", f"{res.get('generalization_degradation', {}).get('A_to_B_mse_ratio', float('nan')):.3f}")
    m4.metric("B→A MSE ratio", f"{res.get('generalization_degradation', {}).get('B_to_A_mse_ratio', float('nan')):.3f}")


if "active_view" not in st.session_state:
    st.session_state.active_view = "Map"

st.session_state.active_view = st.radio(
    "View",
    ["Map", "Feature Inspector", "Cluster Concepts"],
    index=["Map", "Feature Inspector", "Cluster Concepts"].index(st.session_state.active_view),
    horizontal=True,
)

if st.session_state.active_view == "Map":
    c1, c2 = st.columns([3, 1])
    with c1:
        proj = st.radio("Projection", ["2D", "3D"], horizontal=True)
        color_by = st.selectbox("Color by", ["cluster", "selectivity_mag_AminusB", "abs_sel"])

        if proj == "2D":
            fig = px.scatter(
                view,
                x="pc1",
                y="pc2",
                color=(view["cluster"].astype(str) if color_by == "cluster" else view[color_by]),
                hover_name="concept",
                hover_data={"feature": True, "pc1": False, "pc2": False, "pc3": False, "cluster": False},
                height=650,
                title=f"{k}/{layer} · {len(view)} features",
            )
        else:
            fig = px.scatter_3d(
                view,
                x="pc1",
                y="pc2",
                z="pc3",
                color=(view["cluster"].astype(str) if color_by == "cluster" else view[color_by]),
                hover_name="concept",
                hover_data={"feature": True, "pc1": False, "pc2": False, "pc3": False, "cluster": False},
                height=650,
                title=f"{k}/{layer} · {len(view)} features",
            )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Most interesting features (select row to open inspector)**")
        top = view.sort_values("abs_sel", ascending=False).head(20)
        ev = st.dataframe(
            top[["feature", "cluster", "concept", "selectivity_mag_AminusB"]],
            height=300,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key="top_table_select",
        )
        if ev and ev.get("selection", {}).get("rows"):
            ridx = ev["selection"]["rows"][0]
            if 0 <= ridx < len(top):
                st.session_state.selected_feature = int(top.iloc[ridx]["feature"])
                st.session_state.active_view = "Feature Inspector"
                st.rerun()

        st.markdown("**Quick interpretation of axes**")
        st.caption("PC axes are similarity coordinates, not literal concepts.")

elif st.session_state.active_view == "Feature Inspector":
    default_f = int(view.sort_values("abs_sel", ascending=False).feature.iloc[0]) if len(view) else int(df.feature.iloc[0])
    current = int(st.session_state.selected_feature) if st.session_state.selected_feature is not None else default_f
    fid = int(st.number_input("Feature ID", min_value=0, max_value=int(df.feature.max()), value=current, key="feature_id_input"))
    st.session_state.selected_feature = fid
    row = df[df.feature == fid]
    if row.empty:
        st.warning("Feature not found")
    else:
        r = row.iloc[0]
        a, b = fa.get(str(fid), {}), fb.get(str(fid), {})
        st.write(
            {
                "concept": r.concept,
                "cluster": int(r.cluster),
                "selectivity_mag_AminusB": float(r.selectivity_mag_AminusB),
                "freqA": float(r.freqA),
                "freqB": float(r.freqB),
                "label_a": r.label_a,
                "label_b": r.label_b,
            }
        )

        ndf = nearest(df, fid, n=10)
        st.markdown("**Nearest features (select row to inspect it)**")
        evn = st.dataframe(
            ndf[["feature", "cluster", "distance", "concept", "selectivity_mag_AminusB"]],
            height=260,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key="nearest_table_select",
        )
        if evn and evn.get("selection", {}).get("rows"):
            ridx = evn["selection"]["rows"][0]
            if 0 <= ridx < len(ndf):
                st.session_state.selected_feature = int(ndf.iloc[ridx]["feature"])
                st.rerun()

        ca, cb = st.columns(2)
        with ca:
            st.markdown("### Domain A contexts")
            st.write(f"Label: {a.get('heuristic_label', 'n/a')}")
            for c in (a.get("top_contexts", []) or [])[:6]:
                st.caption(c[:300])
        with cb:
            st.markdown("### Domain B contexts")
            st.write(f"Label: {b.get('heuristic_label', 'n/a')}")
            for c in (b.get("top_contexts", []) or [])[:6]:
                st.caption(c[:300])

else:
    cl = cluster_labels(df)
    st.dataframe(cl, use_container_width=True, height=300)
    cid = int(st.selectbox("Inspect cluster", cl["cluster"].astype(int).tolist()))
    cg = df[df.cluster == cid].sort_values("abs_sel", ascending=False).head(30)
    st.markdown("**Representative features in cluster (select row to open inspector)**")
    evc = st.dataframe(
        cg[["feature", "concept", "label_a", "label_b", "selectivity_mag_AminusB", "snippet"]],
        use_container_width=True,
        height=340,
        on_select="rerun",
        selection_mode="single-row",
        key="cluster_table_select",
    )
    if evc and evc.get("selection", {}).get("rows"):
        ridx = evc["selection"]["rows"][0]
        if 0 <= ridx < len(cg):
            st.session_state.selected_feature = int(cg.iloc[ridx]["feature"])
            st.session_state.active_view = "Feature Inspector"
            st.rerun()

st.markdown("---")
st.caption(f"Loaded from: {base}")
