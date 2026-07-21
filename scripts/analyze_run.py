import os
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
from polylex_chatbot.constants import EVALUATION_DATASET_NAME, EVALUATION_RESULTS_PATH, COLS_ORDER_IN_EVALUATION_DF, DICT_METRIC_LABELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

def create_analysis_results_dir(parent_path, llm_name):
    analysis_results_dir = parent_path / f"{llm_name}"
    analysis_results_dir.mkdir(parents=True, exist_ok=True)
    return analysis_results_dir

def create_df_from_langfuse_traces(results):
    rows = []
    for result in results:
        question = result.input.get("query")
        generated_answer = result.output.get("generated_response")
        for metric in result.scores:
            rows.append({
                "trace_id": metric.observation_id,
                "question": question,
                "generated_answer": generated_answer,
                "metric": metric.name,
                "value": metric.value
            })
    df_long = pd.DataFrame(rows)
    df_wide = (
        df_long
        .pivot_table(
            index=["trace_id", "question", "generated_answer"],
            columns=["metric"],
            values="value",
            aggfunc="first"
        )
        .reset_index()
    )
    df_wide.columns.name = None
    return df_wide

def create_df_from_langfuse_run(langfuse, run):
    trace_ids = [item.trace_id for item in run.dataset_run_items]
    results = [langfuse.api.trace.get(trace_id) for trace_id in trace_ids]
    df_results = create_df_from_langfuse_traces(results)
    df_results_ordered = df_results[[col for col in COLS_ORDER_IN_EVALUATION_DF if col in df_results.columns]]
    return df_results_ordered

def validate_scores(df_results):
    df_results = df_results.copy()
    score_columns = df_results.drop(columns=["trace_id", "question", "generated_answer"])
    are_scores_in_range = ((score_columns >= 0) & (score_columns <= 1)).all().all()
    if not are_scores_in_range:
        # a bit bad but LLM sometimes hallucinate
        if ((df_results["trace_id"] == "ec0b9ac77f760564") & (df_results["Answer Correctness - RAGAS"] == 100.0)).any():
            df_results.loc[df_results["trace_id"] == "ec0b9ac77f760564", "Answer Correctness - RAGAS"] = 0.6
        elif ((df_results["trace_id"] == "b925f2da614ac791") & (df_results["Answer Correctness - RAGAS"] == 100.0)).any():
            df_results.loc[df_results["trace_id"] == "b925f2da614ac791", "Answer Correctness - RAGAS"] = 0.2
        elif ((df_results["trace_id"] == "8dd6b4ff3787634e") & (df_results["Answer Correctness - RAGAS"] == 100.0)).any():
            df_results.loc[df_results["trace_id"] == "8dd6b4ff3787634e", "Answer Correctness - RAGAS"] = 1.0
        elif ((df_results["trace_id"] == "0414c474f0ff1dcd") & (df_results["Answer Correctness - RAGAS"] == 100.0)).any():
            df_results.loc[df_results["trace_id"] == "0414c474f0ff1dcd", "Answer Correctness - RAGAS"] = 1.0
        elif ((df_results["trace_id"] == "e181c5c7c0876fc3") & (df_results["Answer Correctness - RAGAS"] == 100.0)).any():
            df_results.loc[df_results["trace_id"] == "e181c5c7c0876fc3", "Answer Correctness - RAGAS"] = 0.9
        elif ((df_results["trace_id"] == "3636db86df87e12c") & (df_results["Answer Correctness - RAGAS"] == 100.0)).any():
            df_results.loc[df_results["trace_id"] == "3636db86df87e12c", "Answer Correctness - RAGAS"] = 1.0
        elif ((df_results["trace_id"] == "00c586f5c3791d51") & (df_results["Groundedness (Faithfulness-RAGAS)"] == 11.0)).any():
            df_results.loc[df_results["trace_id"] == "00c586f5c3791d51", "Groundedness (Faithfulness-RAGAS)"] = 0.9
        elif ((df_results["trace_id"] == "da6effecfdfe26d6") & (df_results["Answer Correctness - RAGAS"] == 100.0)).any():
            df_results.loc[df_results["trace_id"] == "da6effecfdfe26d6", "Answer Correctness - RAGAS"] = 1.0
        elif ((df_results["trace_id"] == "7f9dbece56332599") & (df_results["Answer Correctness - RAGAS"] == 100.0)).any():
            df_results.loc[df_results["trace_id"] == "7f9dbece56332599", "Answer Correctness - RAGAS"] = 1.0
        elif ((df_results["trace_id"] == "15cb148b267b6bf1") & (
                df_results["Answer Correctness - RAGAS"] == 100.0)).any():
            df_results.loc[df_results["trace_id"] == "15cb148b267b6bf1", "Answer Correctness - RAGAS"] = 0.9
        elif ((df_results["trace_id"] == "5f6f528a7e2fd661") & (
                df_results["Answer Correctness - RAGAS"] == 100.0)).any():
            df_results.loc[df_results["trace_id"] == "5f6f528a7e2fd661", "Answer Correctness - RAGAS"] = 1.0
        elif ((df_results["trace_id"] == "a75bf78f28d75822") & (
                df_results["Answer Correctness - RAGAS"] == 100.0)).any():
            df_results.loc[df_results["trace_id"] == "a75bf78f28d75822", "Answer Correctness - RAGAS"] = 0.9
        else:
            raise ValueError("Scores need to be manually checked (out of range)!")
    return df_results

def get_existing_cols(df_cols, cols, context):
    existing_cols = [col for col in cols if col in df_cols]
    missing_cols = [col for col in cols if col not in df_cols]
    if missing_cols:
        logger.info("%s: missing cols (%s)", context, missing_cols)
    return existing_cols

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
        context = f"Kendall matrix for {name}"
        existing_cols = get_existing_cols(df_scores.columns, cols, context)
        scores = df_scores[existing_cols]
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
    plt.figure(figsize=(8, 6))
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

    df_stats_filtred = df_stats.drop(columns=["hit_at_1", "hit_at_2", "hit_at_3", "hit_at_4", "hit_at_5", "hit_at_10", "hit_at_15", "hit_at_20", "ratio_correct_docs"])
    cols_reordered = ["mrr_doc", "Context Relevance (Contextrelevance-Langfuse)", "Groundedness (Faithfulness-RAGAS)", "Answer Relevance (Relevance-Langfuse)", "Answer Correctness - RAGAS", "semantic_similarity", "len_answers_quality", "chrf_score"]
    existing_cols = get_existing_cols(df_stats_filtred.columns, cols_reordered, "Df with all metrics")

    df_stats_reordered = df_stats_filtred[existing_cols]
    plot_statistics(df_stats_reordered, path)

def generate_boxplots_grid(df_results, cols, path):
    existing_cols = get_existing_cols(df_results.columns, cols, "Metric boxplots")
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    axes = axes.flatten()
    legend_handles = None
    legend_labels = None
    hue_order = pd.unique(df_results["question"])
    palette = sns.color_palette("husl", n_colors=len(hue_order))
    for ax, col in zip(axes, existing_cols):
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
    for ax in axes[len(existing_cols):]:
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

def analyze_run(evaluation_dir, dataset_name, run_name, name_dir):
    logger.info("Init Langfuse client...")
    langfuse = Langfuse()
    run = langfuse.get_dataset_run(dataset_name=dataset_name, run_name=run_name)

    if name_dir is None:
        name_dir = run.metadata.get("llm_name").split("/")[1]

    analysis_results_dir = create_analysis_results_dir(evaluation_dir, name_dir)
    logger.info("Directory '%s' created", analysis_results_dir)

    df_results = create_df_from_langfuse_run(langfuse, run)
    df_results = validate_scores(df_results)
    df_results_path = analysis_results_dir / "df_scores_ordered.csv"
    df_results.to_csv(df_results_path, index=False)
    logger.info("Df with all metrics saved in '%s'", df_results_path)

    generate_recall_metrics_plot(df_results, analysis_results_dir)
    logger.info("Plot with recall metrics generated and saved")

    df_scores = df_results.drop(columns=["trace_id", "question", "generated_answer"])

    kendall_groups = {
        "retrieval": ["mrr_doc", "ratio_correct_docs", "Context Relevance (Contextrelevance-Langfuse)"],
        "ground_truth_vs_generated": ["semantic_similarity", "Answer Correctness - RAGAS", "len_answers_quality", "chrf_score"],
        "triad": ["Context Relevance (Contextrelevance-Langfuse)", "Groundedness (Faithfulness-RAGAS)", "Answer Relevance (Relevance-Langfuse)"]
    }
    compute_kendall_matrices(df_scores, kendall_groups, analysis_results_dir)
    logger.info("Kendall matrices computed and saved")

    compute_and_plot_statistics(df_scores, analysis_results_dir)
    logger.info("Statistics computed, plotted and saved")

    metrics_to_plot = ["hit_at_5", "Context Relevance (Contextrelevance-Langfuse)",
                       "Answer Correctness - RAGAS", "semantic_similarity",
                       "len_answers_quality", "Answer Relevance (Relevance-Langfuse)",
                       "Groundedness (Faithfulness-RAGAS)", "chrf_score"]
    generate_boxplots_grid(df_results, metrics_to_plot, analysis_results_dir)
    logger.info("Boxplots grid with selected metrics generated and saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute statistics and save plots from evaluation results")
    parser.add_argument("--env-path", required=True, help="Path to the environment file")
    parser.add_argument("--evaluation-dir", help="Parent directory where results will be stored", default=None)
    parser.add_argument("--dataset-name", help="Dataset on which run has been triggered", default=EVALUATION_DATASET_NAME)
    parser.add_argument("--run-name", required=True, help="Name of the run to analyze")
    parser.add_argument("--name-dir", required=True, help="Name of directory where results will be stored")
    args = parser.parse_args()

    env_file = load_project_env(args.env_path)

    corpus_name = os.getenv("CORPUS_NAME")
    collection_name = os.getenv("DB_COLLECTION_NAME")
    evaluation_dir = args.evaluation_dir or EVALUATION_RESULTS_PATH / corpus_name / collection_name

    analyze_run(
        evaluation_dir=evaluation_dir,
        dataset_name=args.dataset_name,
        run_name=args.run_name,
        name_dir=args.name_dir
    )
