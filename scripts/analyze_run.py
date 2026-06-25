import re
import logging
import textwrap
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from langfuse import Langfuse
import matplotlib.pyplot as plt

from polylex_chatbot.env import load_project_env
env_path = load_project_env()

from polylex_chatbot.config import EVALUATION_DATASET_NAME, EVALUATION_RESULTS_PATH, COLS_ORDER_IN_EVALUATION_DF, DICT_METRIC_LABELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

def create_results_dir(collection_name, llm_name):
    run_path = EVALUATION_RESULTS_PATH / "rag_experiment" / f"{collection_name}_{llm_name}"
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path

def create_df_from_langfuse_scores(results):
    rows = []
    for result in results:
        question = result.input.get("query")
        for metric in result.scores:
            rows.append({
                "trace_id": metric.observation_id,
                "question": question,
                "metric": metric.name,
                "value": metric.value
            })
    df_long = pd.DataFrame(rows)
    df_wide = (
        df_long
        .pivot_table(
            index=["trace_id", "question"],
            columns=["metric"],
            values="value",
            aggfunc="first"
        )
        .reset_index()
    )
    df_wide.columns.name = None
    return df_wide

def create_df_from_langfuse_run(run):
    trace_ids = [item.trace_id for item in run.dataset_run_items]
    results = [langfuse.api.trace.get(trace_id) for trace_id in trace_ids]
    df_results = create_df_from_langfuse_scores(results)
    df_results_ordered = df_results[[col for col in COLS_ORDER_IN_EVALUATION_DF if col in df_results.columns]]
    return df_results_ordered

def validate_scores(df_results):
    df_results = df_results.copy()
    score_columns = df_results.drop(columns=["trace_id", "question"])
    are_scores_in_range = ((score_columns >= 0) & (score_columns <= 1)).all().all()
    if not are_scores_in_range:
        # TODO : faire autrement ou supprimer
        if df_results.loc[df_results["trace_id"] == "ec0b9ac77f760564", "Answer Correctness - RAGAS"].iloc[0] == 100.0:
            df_results.loc[df_results["trace_id"] == "ec0b9ac77f760564", "Answer Correctness - RAGAS"] = 0.6
        else:
            raise ValueError("Scores need to be manually checked (out of range)!")
    return df_results

def generate_recall_metrics_plot(df_results, path):
    recall_cols = [col for col in df_results.columns if re.match(r"^hit_at_\d+$", col)]
    recall_df = pd.DataFrame({
        "k": [int(col.split("_")[-1]) for col in recall_cols],
        "mean_recall": [df_results[col].mean() for col in recall_cols],
    }).sort_values("k")
    plt.figure(figsize=(7, 4))
    plt.plot(
        recall_df["k"],
        recall_df["mean_recall"],
        marker="o",
    )
    plt.xlabel("k")
    plt.ylabel("Recall@k")
    plt.title(f"Mean recall@k on '{collection_name}' (dev dataset)")
    plt.xticks(recall_df["k"])
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.savefig(path / "mean_recall_at_k.png")

def compute_kendall_matrices(df_scores, groups, parent_path):
    for name, cols in groups.items():
        path = parent_path / f"kendall_matrix_{name}.csv"
        scores = df_scores[cols]
        kendall_matrix = scores.corr(method="kendall")
        kendall_matrix.to_csv(path, index=False)

def compute_statistics(df_scores):
    confidence_level = 0.95
    n = len(df_scores)
    means = df_scores.mean()
    stds = df_scores.std(ddof=1)
    ci_lower, ci_upper = stats.t.interval(confidence_level, df=n - 1, loc=means, scale=stds / np.sqrt(n))
    ci_lower = pd.Series(ci_lower, index=df_scores.columns)
    ci_upper = pd.Series(ci_upper, index=df_scores.columns)
    df_stats = pd.DataFrame([means, stds, ci_lower, ci_upper], index=["mean", "standard_deviation", "ci_lower_bound", "ci_upper_bound"])
    return df_stats

def plot_statistics(df_stats, path):
    means = df_stats.loc["mean"]
    ci_lower = df_stats.loc["ci_lower_bound"]
    ci_upper = df_stats.loc["ci_upper_bound"]
    yerr_lower = means - ci_lower
    yerr_upper = ci_upper - means
    x = np.arange(len(df_stats.columns))
    x_labels = [DICT_METRIC_LABELS[col] for col in df_stats.columns]
    plt.figure(figsize=(12, 6))
    plt.errorbar(x, means.values, yerr=[yerr_lower, yerr_upper], fmt="o", capsize=5, linewidth=1.5)
    plt.xticks(x, x_labels, rotation=45, ha="right")
    plt.ylabel("Value")
    plt.title("Mean with CI for each metric")
    plt.axhline(0, linestyle="--", linewidth=0.8)
    plt.axhline(1, linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(path/ "confidence_intervals.png")

def compute_and_plot_statistics(df_scores, path):
    df_stats = compute_statistics(df_scores)
    df_stats.to_csv(path / "df_stats.csv", index=True)
    plot_statistics(df_stats, path)

def generate_boxplots_grid(df_results, cols, path):
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    axes = axes.flatten()
    legend_handles = None
    legend_labels = None
    hue_order = pd.unique(df_results["question"])
    palette = sns.color_palette("husl", n_colors=len(hue_order))
    for ax, col in zip(axes, cols):
        sns.boxplot(data=df_results, y=col, color="lightgray", showfliers=False, width=0.6, ax=ax)
        sns.stripplot(data=df_results, y=col, hue="question", hue_order=hue_order, palette=palette, jitter=0.3, ax=ax)
        ax.set_title(f"{DICT_METRIC_LABELS[col]}")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_ylim(-0.05, 1.05)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    for ax in axes[len(cols):]:
        ax.remove()
    fig.subplots_adjust(bottom=0.26, hspace=0.4, wspace=0.2)
    legend_labels = [
        textwrap.fill(label, width=300)
        for label in legend_labels
    ]
    fig.legend(
        legend_handles,
        legend_labels,
        title="Question",
        loc="lower center",
        bbox_to_anchor=(0.05, 0.02, 0.90, 0.22),
        title_fontsize=10,
        fontsize=8
    )
    plt.savefig(path / "metric_boxplots.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute statistics, generate plots, and save the results.")
    parser.add_argument("run_name", help="Name of the run to analyze")
    parser.add_argument("--dataset-name", default=EVALUATION_DATASET_NAME)
    args = parser.parse_args()

    langfuse = Langfuse()

    logger.info("Collecting metadata from run")
    run = langfuse.get_dataset_run(dataset_name=args.dataset_name, run_name=args.run_name)
    collection_name = run.metadata.get("collection_name")
    llm_name = run.metadata.get("llm_name").split("/")[1]

    results_dir = create_results_dir(collection_name, llm_name)
    logger.info("Directory '%s' created", results_dir)

    df_results = create_df_from_langfuse_run(run)
    df_results = validate_scores(df_results)
    df_results_path = results_dir / "df_scores_ordered.csv"
    df_results.to_csv(df_results_path, index=False)
    logger.info("Df with all metrics saved in '%s'", df_results_path)

    generate_recall_metrics_plot(df_results, results_dir)
    logger.info("Plot with recall metrics generated and saved")

    df_scores = df_results.drop(columns=["trace_id", "question"])

    kendall_groups = {
        "retrieval": ["mrr_doc", "ratio_correct_docs", "Context Relevance (Contextrelevance-Langfuse)"],
        "ground_truth_vs_generated": ["semantic_similarity", "Answer Correctness - RAGAS", "len_answers_quality", "chrf_score"],
        "triad": ["Context Relevance (Contextrelevance-Langfuse)", "Groundedness (Faithfulness-RAGAS)", "Answer Relevance (Relevance-Langfuse)"]
    }
    compute_kendall_matrices(df_scores, kendall_groups, results_dir)
    logger.info("Kendall matrices computed and saved")

    compute_and_plot_statistics(df_scores, results_dir)
    logger.info("Statistics computed, plotted and saved")

    metrics_to_plot = ["hit_at_5", "Context Relevance (Contextrelevance-Langfuse)",
                       "Answer Correctness - RAGAS", "semantic_similarity",
                       "len_answers_quality", "Answer Relevance (Relevance-Langfuse)",
                       "Groundedness (Faithfulness-RAGAS)", "chrf_score"]
    generate_boxplots_grid(df_results, metrics_to_plot, results_dir)
    logger.info("Boxplots grid with selected metrics generated and saved")
