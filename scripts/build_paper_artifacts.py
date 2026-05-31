from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "MAG_Dataset"
DERIVED = ROOT / "data" / "derived"
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"
N_PAPERS = 736_389
N_FIELDS = 59_965
MAX_TOPICS = 6
ANALYTIC_START = 2011
ANALYTIC_END = 2016


def read_vector(path: Path, dtype: np.dtype) -> np.ndarray:
    return np.loadtxt(path, dtype=dtype, delimiter=",")


def read_edges(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.int32, delimiter=",")


def make_dirs() -> None:
    for path in (DERIVED, FIGURES, TABLES):
        path.mkdir(parents=True, exist_ok=True)


def load_topic_csr() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = read_edges(DATA / "raw/relations/paper___has_topic___field_of_study/edge.csv")
    order = np.argsort(edges[:, 0], kind="mergesort")
    paper_ids = edges[order, 0]
    field_ids = edges[order, 1]
    counts = np.bincount(paper_ids, minlength=N_PAPERS).astype(np.int16)
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    return field_ids, offsets, counts


def paper_topics(field_ids: np.ndarray, offsets: np.ndarray, paper_id: int) -> np.ndarray:
    start = offsets[paper_id]
    end = offsets[paper_id + 1]
    topics = np.unique(field_ids[start:end])
    if topics.size > MAX_TOPICS:
        topics = topics[:MAX_TOPICS]
    return topics


def gf2_rank(matrix: np.ndarray) -> int:
    if matrix.size == 0:
        return 0
    a = matrix.copy().astype(np.uint8) % 2
    rows, cols = a.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for row in range(rank, rows):
            if a[row, col]:
                pivot = row
                break
        if pivot is None:
            continue
        if pivot != rank:
            a[[rank, pivot]] = a[[pivot, rank]]
        for row in range(rows):
            if row != rank and a[row, col]:
                a[row] ^= a[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def local_complex_stats(topics: np.ndarray, existing_edges: set[tuple[int, int]]) -> tuple[int, int, int, float, float, float]:
    n = int(topics.size)
    if n == 0:
        return 0, 0, 0, 0.0, 0.0, 0.0
    local_index = {int(topic): pos for pos, topic in enumerate(topics.tolist())}
    edges = [(local_index[u], local_index[v]) for u, v in existing_edges]
    triangles = []
    tetrahedra = []
    for a, b, c in combinations(range(n), 3):
        if (a, b) in edges and (a, c) in edges and (b, c) in edges:
            triangles.append((a, b, c))
    for a, b, c, d in combinations(range(n), 4):
        faces = ((a, b, c), (a, b, d), (a, c, d), (b, c, d))
        if all(face in triangles for face in faces):
            tetrahedra.append((a, b, c, d))
    d1 = np.zeros((n, len(edges)), dtype=np.uint8)
    for col, (u, v) in enumerate(edges):
        d1[u, col] = 1
        d1[v, col] = 1
    edge_index = {edge: pos for pos, edge in enumerate(edges)}
    d2 = np.zeros((len(edges), len(triangles)), dtype=np.uint8)
    for col, (a, b, c) in enumerate(triangles):
        for edge in ((a, b), (a, c), (b, c)):
            d2[edge_index[edge], col] = 1
    triangle_index = {tri: pos for pos, tri in enumerate(triangles)}
    d3 = np.zeros((len(triangles), len(tetrahedra)), dtype=np.uint8)
    for col, (a, b, c, d) in enumerate(tetrahedra):
        for face in ((a, b, c), (a, b, d), (a, c, d), (b, c, d)):
            d3[triangle_index[face], col] = 1
    rank_d1 = gf2_rank(d1)
    rank_d2 = gf2_rank(d2)
    rank_d3 = gf2_rank(d3)
    beta0 = n - rank_d1
    beta1 = len(edges) - rank_d1 - rank_d2
    beta2 = len(triangles) - rank_d2 - rank_d3
    adjacency = np.zeros((n, n), dtype=float)
    for u, v in edges:
        adjacency[u, v] = 1.0
        adjacency[v, u] = 1.0
    degree = adjacency.sum(axis=1)
    laplacian = np.diag(degree) - adjacency
    eigvals = np.linalg.eigvalsh(laplacian)
    lambda2 = float(eigvals[1]) if n > 1 else 0.0
    adjacency_eigs = np.linalg.eigvalsh(adjacency) if n > 0 else np.array([0.0])
    spectral_radius = float(np.max(np.abs(adjacency_eigs))) if adjacency_eigs.size else 0.0
    positive = eigvals[eigvals > 1e-12]
    if positive.size and positive.sum() > 0:
        probs = positive / positive.sum()
        spectral_entropy = float(-(probs * np.log(probs)).sum())
    else:
        spectral_entropy = 0.0
    return int(beta0), int(beta1), int(beta2), lambda2, spectral_radius, spectral_entropy


def future_citations(citation_edges: np.ndarray, years: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_year = years[citation_edges[:, 0]]
    target_year = years[citation_edges[:, 1]]
    future_mask = (source_year > target_year) & ((source_year - target_year) <= 3)
    future3 = np.bincount(citation_edges[future_mask, 1], minlength=N_PAPERS).astype(np.int32)
    references = np.bincount(citation_edges[:, 0], minlength=N_PAPERS).astype(np.int16)
    return future3, references


def score_paper(topics: np.ndarray, pair_set: set[int], degree: np.ndarray) -> tuple[float, float, float, float, int, int, int, float, float, float]:
    if topics.size < 2:
        beta0, beta1, beta2, lambda2, radius, entropy = local_complex_stats(topics, set())
        return 0.0, 0.0, 0.0, 0.0, beta0, beta1, beta2, lambda2, radius, entropy
    pairs = []
    novel = 0
    curvatures = []
    prior_degree = []
    existing_edges: set[tuple[int, int]] = set()
    for u_raw, v_raw in combinations(topics.tolist(), 2):
        u, v = sorted((int(u_raw), int(v_raw)))
        key = u * N_FIELDS + v
        exists = key in pair_set
        pairs.append((u, v, exists))
        if exists:
            curvatures.append(4 - int(degree[u]) - int(degree[v]))
            existing_edges.add((u, v))
        else:
            novel += 1
        prior_degree.extend((int(degree[u]), int(degree[v])))
    novel_share = novel / len(pairs)
    if topics.size >= 3:
        existing_pairs = {(u, v) for u, v, exists in pairs if exists}
        boundary_hits = 0
        boundary_total = 0
        for a_raw, b_raw, c_raw in combinations(topics.tolist(), 3):
            a, b, c = sorted((int(a_raw), int(b_raw), int(c_raw)))
            boundary_hits += ((a, b) in existing_pairs) and ((a, c) in existing_pairs) and ((b, c) in existing_pairs)
            boundary_total += 1
        boundary_share = boundary_hits / boundary_total if boundary_total else 0.0
    else:
        boundary_share = 0.0
    mean_curvature = float(np.mean(curvatures)) if curvatures else 0.0
    mean_prior_degree = float(np.mean(prior_degree)) if prior_degree else 0.0
    beta0, beta1, beta2, lambda2, radius, entropy = local_complex_stats(topics, existing_edges)
    return novel_share, boundary_share, mean_curvature, mean_prior_degree, beta0, beta1, beta2, lambda2, radius, entropy


def update_pairs(topics: np.ndarray, pair_set: set[int], degree: np.ndarray) -> None:
    if topics.size < 2:
        return
    for u_raw, v_raw in combinations(topics.tolist(), 2):
        u, v = sorted((int(u_raw), int(v_raw)))
        key = u * N_FIELDS + v
        if key not in pair_set:
            pair_set.add(key)
            degree[u] += 1
            degree[v] += 1


def build_topology_features(years: np.ndarray, field_ids: np.ndarray, offsets: np.ndarray, topic_counts: np.ndarray) -> pd.DataFrame:
    pair_set: set[int] = set()
    degree = np.zeros(N_FIELDS, dtype=np.int32)
    rows = []
    for year in sorted(np.unique(years)):
        paper_ids = np.where(years == year)[0]
        for paper_id in paper_ids:
            topics = paper_topics(field_ids, offsets, int(paper_id))
            novel, boundary, curvature, prior_degree, beta0, beta1, beta2, lambda2, radius, entropy = score_paper(topics, pair_set, degree)
            rows.append(
                {
                    "paper_id": int(paper_id),
                    "year": int(year),
                    "topic_count": int(topic_counts[paper_id]),
                    "topic_count_capped": int(topics.size),
                    "novel_pair_share": novel,
                    "boundary_completion_share": boundary,
                    "mean_forman_curvature": curvature,
                    "mean_prior_topic_degree": prior_degree,
                    "local_betti_0": beta0,
                    "local_betti_1": beta1,
                    "local_betti_2": beta2,
                    "local_laplacian_lambda2": lambda2,
                    "local_spectral_radius": radius,
                    "local_spectral_entropy": entropy,
                    "prior_topic_edges": len(pair_set),
                }
            )
        for paper_id in paper_ids:
            update_pairs(paper_topics(field_ids, offsets, int(paper_id)), pair_set, degree)
    return pd.DataFrame(rows)


def add_outcomes(features: pd.DataFrame, years: np.ndarray, labels: np.ndarray, future3: np.ndarray, references: np.ndarray, author_count: np.ndarray) -> pd.DataFrame:
    features = features.copy()
    features["label"] = labels[features["paper_id"].to_numpy()]
    features["future_cites_3y"] = future3[features["paper_id"].to_numpy()]
    features["references"] = references[features["paper_id"].to_numpy()]
    features["author_count"] = author_count[features["paper_id"].to_numpy()]
    features["log_future_cites_3y"] = np.log1p(features["future_cites_3y"])
    features["negative_forman_curvature"] = -features["mean_forman_curvature"]
    for col in (
        "novel_pair_share",
        "boundary_completion_share",
        "negative_forman_curvature",
        "local_betti_0",
        "local_betti_1",
        "local_betti_2",
        "local_laplacian_lambda2",
        "local_spectral_radius",
        "local_spectral_entropy",
    ):
        subset = features.loc[(features["year"] >= ANALYTIC_START) & (features["year"] <= ANALYTIC_END), col]
        sd = subset.std(ddof=0)
        features[f"z_{col}"] = (features[col] - subset.mean()) / sd if sd > 0 else 0.0
    features["topological_opportunity"] = features["z_boundary_completion_share"]
    subset_mask = (features["year"] >= ANALYTIC_START) & (features["year"] <= ANALYTIC_END)
    features["breakthrough_top5"] = 0
    for year, group in features.loc[subset_mask].groupby("year"):
        cutoff = group["future_cites_3y"].quantile(0.95)
        idx = group.index[group["future_cites_3y"] >= cutoff]
        features.loc[idx, "breakthrough_top5"] = 1
    return features


def design_matrix(df: pd.DataFrame, covariates: list[str], use_fe: bool = True) -> tuple[np.ndarray, list[str]]:
    parts = [pd.Series(1.0, index=df.index, name="Intercept"), df[covariates].astype(float)]
    if use_fe:
        top_labels = df["label"].value_counts().head(30).index
        venue_group = df["label"].where(df["label"].isin(top_labels), other=-1).astype(str)
        parts.append(pd.get_dummies(df["year"].astype(str), prefix="year", drop_first=True, dtype=float))
        parts.append(pd.get_dummies(venue_group, prefix="venue", drop_first=True, dtype=float))
    xdf = pd.concat(parts, axis=1)
    return xdf.to_numpy(dtype=float), xdf.columns.tolist()


def ols_hc1(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    xr = X * resid[:, None]
    vcov = (n / max(n - k, 1)) * xtx_inv @ (xr.T @ xr) @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    tss = float(np.sum((y - y.mean()) ** 2))
    rss = float(np.sum(resid**2))
    r2 = 1 - rss / tss if tss else 0.0
    return beta, se, r2


def star(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def regression_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = ["log_authors", "log_topics", "log_references"]
    specs = [
        ("Log cites", "Baseline", "log_future_cites_3y", base),
        ("Log cites", "+ Topology index", "log_future_cites_3y", base + ["topological_opportunity"]),
        (
            "Log cites",
            "+ TDA/Spectral",
            "log_future_cites_3y",
            base
            + [
                "z_novel_pair_share",
                "z_local_betti_1",
                "z_local_laplacian_lambda2",
                "z_local_spectral_entropy",
                "z_negative_forman_curvature",
            ],
        ),
        ("Breakthrough", "Baseline", "breakthrough_top5", base),
        ("Breakthrough", "+ Topology index", "breakthrough_top5", base + ["topological_opportunity"]),
        (
            "Breakthrough",
            "+ TDA/Spectral",
            "breakthrough_top5",
            base
            + [
                "z_novel_pair_share",
                "z_local_betti_1",
                "z_local_laplacian_lambda2",
                "z_local_spectral_entropy",
                "z_negative_forman_curvature",
            ],
        ),
    ]
    rows = []
    meta = []
    for outcome_name, spec_name, ycol, covariates in specs:
        X, names = design_matrix(df, covariates)
        y = df[ycol].to_numpy(dtype=float)
        beta, se, r2 = ols_hc1(y, X)
        result = dict(zip(names, zip(beta, se)))
        col = f"{outcome_name}: {spec_name}"
        for key in (
            "topological_opportunity",
            "z_novel_pair_share",
            "z_local_betti_1",
            "z_local_laplacian_lambda2",
            "z_local_spectral_entropy",
            "z_negative_forman_curvature",
        ):
            if key in result:
                coef, stderr = result[key]
                p = 2 * norm.sf(abs(coef / stderr)) if stderr > 0 else 1.0
                rows.append({"variable": key, "model": col, "coef": coef, "se": stderr, "p": p})
        meta.append({"model": col, "n": len(df), "r2": r2, "mean_y": y.mean()})
    return pd.DataFrame(rows), pd.DataFrame(meta)


def write_regression_table(rows: pd.DataFrame, meta: pd.DataFrame) -> None:
    variables = [
        ("topological_opportunity", "TO index"),
        ("z_novel_pair_share", "Novel pairs"),
        ("z_local_betti_1", "$\\beta_1$ holes"),
        ("z_local_laplacian_lambda2", "$\\lambda_2(L)$"),
        ("z_local_spectral_entropy", "Spectral entropy"),
        ("z_negative_forman_curvature", "Negative curvature"),
    ]
    models = meta["model"].tolist()
    lines = []
    lines.append(r"\begin{table}[!htbp]\centering")
    lines.append(r"\begin{threeparttable}")
    lines.append(r"\caption{Topology, geometry, and future scientific impact}")
    lines.append(r"\label{tab:regressions}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l" + "c" * len(models) + r"}")
    lines.append(r"\toprule")
    lines.append(" & " + " & ".join([f"({i + 1})" for i in range(len(models))]) + r" \\")
    lines.append(r"\midrule")
    for key, label in variables:
        coefs = []
        ses = []
        for model in models:
            hit = rows[(rows["variable"] == key) & (rows["model"] == model)]
            if hit.empty:
                coefs.append("")
                ses.append("")
            else:
                item = hit.iloc[0]
                coefs.append(f"{item['coef']:.3f}{star(float(item['p']))}")
                ses.append(f"({item['se']:.3f})")
        lines.append(label + " & " + " & ".join(coefs) + r" \\")
        lines.append(" & " + " & ".join(ses) + r" \\")
    lines.append(r"\midrule")
    outcome_labels = {"Log cites": "Log cites", "Breakthrough": "Top 5\\%"}
    spec_labels = {"Baseline": "Base", "+ Topology index": "+ TO", "+ TDA/Spectral": "+ TDA/Spec."}
    lines.append("Outcome & " + " & ".join([outcome_labels[m.split(":")[0]] for m in models]) + r" \\")
    lines.append("Specification & " + " & ".join([spec_labels[m.split(": ")[1]] for m in models]) + r" \\")
    lines.append("Controls & " + " & ".join(["Yes"] * len(models)) + r" \\")
    lines.append("Year FE & " + " & ".join(["Yes"] * len(models)) + r" \\")
    lines.append("Venue-group FE & " + " & ".join(["Yes"] * len(models)) + r" \\")
    lines.append("$N$ & " + " & ".join([f"{int(n):,}" for n in meta["n"]]) + r" \\")
    lines.append("$R^2$ & " + " & ".join([f"{r2:.3f}" for r2 in meta["r2"]]) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}%")
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\footnotesize")
    lines.append(r"\item Notes: HC1 robust standard errors in parentheses. The analytic sample contains papers published from 2011 through 2016, allowing a three-year forward citation window. Topology variables are standardized within the analytic sample. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{threeparttable}")
    lines.append(r"\end{table}")
    (TABLES / "regression_table.tex").write_text("\n".join(lines) + "\n")


def prediction_table(df: pd.DataFrame) -> pd.DataFrame:
    base = ["log_authors", "log_topics", "log_references"]
    topo = base + [
        "topological_opportunity",
        "z_novel_pair_share",
        "z_local_betti_1",
        "z_local_laplacian_lambda2",
        "z_local_spectral_entropy",
        "z_negative_forman_curvature",
    ]
    rows = []
    for name, covariates in (("Baseline controls", base), ("Controls + topology", topo)):
        X, _ = design_matrix(df, covariates)
        y = df["breakthrough_top5"].to_numpy(dtype=int)
        train = df["year"].to_numpy() <= 2014
        model = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000, class_weight="balanced"))
        model.fit(X[train], y[train])
        pred = model.predict_proba(X[~train])[:, 1]
        rows.append(
            {
                "model": name,
                "auc": roc_auc_score(y[~train], pred),
                "average_precision": average_precision_score(y[~train], pred),
                "test_n": int((~train).sum()),
            }
        )
    return pd.DataFrame(rows)


def write_prediction_table(pred: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[!htbp]\centering",
        r"\begin{threeparttable}",
        r"\caption{Out-of-time prediction of breakthrough papers}",
        r"\label{tab:prediction}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & ROC AUC & Average precision & Test $N$ \\",
        r"\midrule",
    ]
    for _, row in pred.iterrows():
        lines.append(f"{row['model']} & {row['auc']:.3f} & {row['average_precision']:.3f} & {int(row['test_n']):,} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\footnotesize",
            r"\item Notes: Models train on 2011--2014 papers and test on 2015--2016 papers. Breakthrough papers are defined as top-five-percent papers within publication year by three-year forward citations.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
        ]
    )
    (TABLES / "prediction_table.tex").write_text("\n".join(lines) + "\n")


def write_summary_tables(features: pd.DataFrame, analytic: pd.DataFrame, pred: pd.DataFrame, reg_meta: pd.DataFrame) -> None:
    stats = {
        "Papers": N_PAPERS,
        "Authors": 1_134_649,
        "Institutions": 8_740,
        "Fields of study": N_FIELDS,
        "Citation edges": 5_416_271,
        "Paper-topic edges": 7_505_078,
        "Author-paper edges": 7_145_660,
        "Analytic papers": len(analytic),
    }
    lines = [
        r"\begin{table}[!htbp]\centering",
        r"\caption{Dataset and analytic sample}",
        r"\label{tab:data}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Quantity & Count \\",
        r"\midrule",
    ]
    for key, value in stats.items():
        lines.append(f"{key} & {value:,} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (TABLES / "data_table.tex").write_text("\n".join(lines) + "\n")
    summary_cols = [
        "future_cites_3y",
        "breakthrough_top5",
        "novel_pair_share",
        "boundary_completion_share",
        "negative_forman_curvature",
        "topological_opportunity",
        "local_betti_0",
        "local_betti_1",
        "local_betti_2",
        "local_laplacian_lambda2",
        "local_spectral_radius",
        "local_spectral_entropy",
        "author_count",
        "topic_count",
        "references",
    ]
    desc = analytic[summary_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    desc = desc[["mean", "std", "25%", "50%", "75%"]]
    lines = [
        r"\begin{table}[!htbp]\centering",
        r"\caption{Summary statistics}",
        r"\label{tab:summary}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Variable & Mean & Std. dev. & P25 & Median & P75 \\",
        r"\midrule",
    ]
    labels = {
        "future_cites_3y": "Three-year forward citations",
        "breakthrough_top5": "Breakthrough indicator",
        "novel_pair_share": "Novel topic-pair share",
        "boundary_completion_share": "Boundary-completion potential",
        "negative_forman_curvature": "Negative Forman curvature",
        "topological_opportunity": "Topological opportunity index",
        "local_betti_0": "Local $\\beta_0$",
        "local_betti_1": "Local $\\beta_1$",
        "local_betti_2": "Local $\\beta_2$",
        "local_laplacian_lambda2": "Local $\\lambda_2(L)$",
        "local_spectral_radius": "Local spectral radius",
        "local_spectral_entropy": "Local spectral entropy",
        "author_count": "Authors",
        "topic_count": "Topics",
        "references": "References",
    }
    for key, row in desc.iterrows():
        lines.append(f"{labels[key]} & {row['mean']:.3f} & {row['std']:.3f} & {row['25%']:.3f} & {row['50%']:.3f} & {row['75%']:.3f} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    (TABLES / "summary_table.tex").write_text("\n".join(lines) + "\n")
    macros = [
        f"\\newcommand{{\\AnalyticN}}{{{len(analytic):,}}}",
        f"\\newcommand{{\\AucBaseline}}{{{pred.loc[pred['model'] == 'Baseline controls', 'auc'].iloc[0]:.3f}}}",
        f"\\newcommand{{\\AucTopology}}{{{pred.loc[pred['model'] == 'Controls + topology', 'auc'].iloc[0]:.3f}}}",
        f"\\newcommand{{\\RegRtwoBaseline}}{{{reg_meta['r2'].iloc[0]:.3f}}}",
        f"\\newcommand{{\\RegRtwoTopology}}{{{reg_meta['r2'].iloc[1]:.3f}}}",
    ]
    (TABLES / "results_macros.tex").write_text("\n".join(macros) + "\n")


def make_figures(features: pd.DataFrame, analytic: pd.DataFrame, pred: pd.DataFrame) -> None:
    annual = features.groupby("year").agg(
        novel_pair_share=("novel_pair_share", "mean"),
        boundary_completion_share=("boundary_completion_share", "mean"),
        negative_forman_curvature=("negative_forman_curvature", "mean"),
        local_betti_1=("local_betti_1", "mean"),
        local_laplacian_lambda2=("local_laplacian_lambda2", "mean"),
        local_spectral_entropy=("local_spectral_entropy", "mean"),
        prior_topic_edges=("prior_topic_edges", "mean"),
    )
    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
    ax1.plot(annual.index, annual["novel_pair_share"], marker="o", label="Novel topic-pair share")
    ax1.plot(annual.index, annual["boundary_completion_share"], marker="s", label="Boundary-completion potential")
    ax1.set_xlabel("Publication year")
    ax1.set_ylabel("Mean share")
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(annual.index, annual["prior_topic_edges"] / 1_000_000, color="black", linestyle="--", label="Prior topic edges")
    ax2.set_ylabel("Prior topic edges, millions")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(FIGURES / "annual_topology.pdf")
    plt.close(fig)
    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
    ax1.plot(annual.index, annual["local_betti_1"], marker="o", label=r"Mean local $\beta_1$")
    ax1.set_xlabel("Publication year")
    ax1.set_ylabel(r"Mean local $\beta_1$")
    ax2 = ax1.twinx()
    ax2.plot(annual.index, annual["local_laplacian_lambda2"], marker="s", color="darkred", label=r"Mean local $\lambda_2(L)$")
    ax2.plot(annual.index, annual["local_spectral_entropy"], marker="^", color="black", linestyle="--", label="Mean spectral entropy")
    ax2.set_ylabel("Spectral metrics")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "annual_tda_spectral.pdf")
    plt.close(fig)
    temp = analytic.copy()
    temp["topology_bin"] = pd.qcut(temp["topological_opportunity"], 20, labels=False, duplicates="drop")
    binned = temp.groupby("topology_bin").agg(
        topological_opportunity=("topological_opportunity", "mean"),
        future_cites_3y=("future_cites_3y", "mean"),
        breakthrough_top5=("breakthrough_top5", "mean"),
    )
    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
    ax1.plot(binned["topological_opportunity"], binned["future_cites_3y"], marker="o", label="Forward citations")
    ax1.set_xlabel("Topological opportunity index, ventiles")
    ax1.set_ylabel("Mean three-year forward citations")
    ax2 = ax1.twinx()
    ax2.plot(binned["topological_opportunity"], binned["breakthrough_top5"], marker="s", color="darkred", label="Breakthrough probability")
    ax2.set_ylabel("Breakthrough probability")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "topology_binscatter.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = np.arange(len(pred))
    ax.bar(x - 0.18, pred["auc"], width=0.36, label="ROC AUC")
    ax.bar(x + 0.18, pred["average_precision"], width=0.36, label="Average precision")
    ax.set_xticks(x)
    ax.set_xticklabels(pred["model"], rotation=10, ha="right")
    ax.set_ylim(0, max(0.8, pred[["auc", "average_precision"]].to_numpy().max() + 0.08))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "prediction_performance.pdf")
    plt.close(fig)


def main() -> None:
    make_dirs()
    years = read_vector(DATA / "raw/node-feat/paper/node_year.csv", np.int16)
    labels = read_vector(DATA / "raw/node-label/paper/node-label.csv", np.int16)
    field_ids, offsets, topic_counts = load_topic_csr()
    citation_edges = read_edges(DATA / "raw/relations/paper___cites___paper/edge.csv")
    author_edges = read_edges(DATA / "raw/relations/author___writes___paper/edge.csv")
    future3, references = future_citations(citation_edges, years)
    author_count = np.bincount(author_edges[:, 1], minlength=N_PAPERS).astype(np.int16)
    features = build_topology_features(years, field_ids, offsets, topic_counts)
    features = add_outcomes(features, years, labels, future3, references, author_count)
    features["log_authors"] = np.log1p(features["author_count"])
    features["log_topics"] = np.log1p(features["topic_count"])
    features["log_references"] = np.log1p(features["references"])
    features.to_csv(DERIVED / "paper_features.csv", index=False)
    analytic = features[
        (features["year"] >= ANALYTIC_START)
        & (features["year"] <= ANALYTIC_END)
        & (features["topic_count_capped"] >= 2)
    ].copy()
    rows, reg_meta = regression_models(analytic)
    pred = prediction_table(analytic)
    write_regression_table(rows, reg_meta)
    write_prediction_table(pred)
    write_summary_tables(features, analytic, pred, reg_meta)
    make_figures(features, analytic, pred)
    print(f"Wrote analysis artifacts for {len(analytic):,} analytic papers.")


if __name__ == "__main__":
    main()
