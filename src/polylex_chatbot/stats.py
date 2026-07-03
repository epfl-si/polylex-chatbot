import os
import json
import pandas as pd
from tika import parser
from docx import Document
from collections import Counter
from matplotlib import pyplot as plt
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from polylex_chatbot.config import ARTICLE_PATTERN

def count_nb_summaries(data):
    nb_summaries = 0
    for metadata_dict in data.values():
        nb_summaries += len(metadata_dict.get("summaries"))
    return nb_summaries

def count_duplicated_docs(data):
    nb_duplicated_docs, nb_mismatch_categories, nb_mismatch_languages = 0, 0, 0

    for metadata_dict in data.values():
        refs = metadata_dict.get("refs")
        if len(refs) > 1:
            nb_duplicated_docs += 1
            cats = metadata_dict.get("cats")
            if len(cats) > 1:
                # doc is in Polylex index and mentioned as appendix
                nb_mismatch_categories += 1
            else:
                # doc is only available in one language
                nb_mismatch_languages += 1

    return nb_duplicated_docs, nb_mismatch_categories, nb_mismatch_languages

def count_per_lang_and_key(data, key):
    counter = Counter()

    for metadata_dict in data.values():
        lang = metadata_dict.get("lang")
        key_content = metadata_dict.get(key)
        if isinstance(key_content, str):
            key_content = (key_content,)
        else:
            key_content = tuple(sorted(metadata_dict.get(key)))
        counter_key = (lang, key_content)
        counter[counter_key] += 1

    result = {
        f"nb_{lang}_{'_'.join(key_content)}": count
        for (lang, key_content), count in counter.items()
    }

    return result

def count_ratio_alnum_chars(content):
    content_length = len(content) if len(content) > 0 else 1
    nb_alnum_chars, nb_al_chars = 0, 0

    for char in content:
        if char.isalnum():
            nb_alnum_chars += 1
            if char.isalpha():
                nb_al_chars += 1

    ratio_alnum_chars = nb_alnum_chars / content_length
    ratio_al_chars = nb_al_chars / content_length
    ratio_num_chars = (nb_alnum_chars - nb_al_chars) / content_length

    return ratio_alnum_chars, ratio_al_chars, ratio_num_chars

def count_nb_tokens(content):
    model_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    tokenizer = MistralTokenizer.from_hf_hub(model_id)
    nb_tokens = len(tokenizer.instruct_tokenizer.tokenizer.encode(content, bos=False, eos=False))
    return nb_tokens

def compute_corpus_metadata_stats(data):
    corpus_metadata_stats = {
        "corpus_size": len(data),
        "nb_summaries": count_nb_summaries(data)
    }

    languages_result = Counter(item["lang"] for item in data.values())
    for lang, count in languages_result.items():
        corpus_metadata_stats[f"nb_{lang}"] = count

    cats_result = count_per_lang_and_key(data, "cats")
    for key, count in cats_result.items():
        corpus_metadata_stats[key] = count

    sources_result = count_per_lang_and_key(data, "source")
    for key, count in sources_result.items():
        corpus_metadata_stats[key] = count

    content_formats_result = count_per_lang_and_key(data, "content_format")
    for key, count in content_formats_result.items():
        corpus_metadata_stats[key] = count

    corpus_metadata_stats["nb_duplicated_docs"], corpus_metadata_stats["nb_mismatch_categories"], corpus_metadata_stats["nb_mismatch_languages"] = count_duplicated_docs(data)

    return corpus_metadata_stats

def compute_file_content_stats(filename, suffix, content):
    nb_occ_article = sum(1 for _ in ARTICLE_PATTERN.finditer(content))
    ratio_alnum, ratio_al, ratio_num = count_ratio_alnum_chars(content)

    stats = {
        "doc_id": filename,
        "suffix": suffix,
        "nb_chars": len(content),
        "nb_tokens": count_nb_tokens(content),
        "ratio_alnum": ratio_alnum,
        "ratio_al": ratio_al,
        "ratio_num": ratio_num,
        "nb_occ_article": nb_occ_article
    }

    return stats

def compute_corpus_content_stats(data_path):
    stats = []

    for file in data_path.iterdir():
        suffix = file.suffix
        if suffix == ".txt":
            content = file.read_text(encoding="utf-8")
            stats.append(compute_file_content_stats(file.name, suffix, content))
        elif suffix == ".docx":
            doc = Document(file)
            content = "\n".join(p.text for p in doc.paragraphs)
            stats.append(compute_file_content_stats(file.name, suffix, content))
        elif suffix == ".pdf":
            parsed = parser.from_file(str(file))
            content = parsed.get("content")
            stats.append(compute_file_content_stats(file.name, suffix, content))
        else:
            # TODO : gerer dans les logs
            print(f"Error while reading {file}: format '{suffix}' not supported")

    return pd.DataFrame(stats)

def save_stats(path, filename, suffix, stats):
    filepath = os.path.join(path, f"{filename}.{suffix}")

    if suffix == "json":
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)
    elif suffix == "csv":
        stats.to_csv(filepath, index=True)

    return filepath

def compute_content_lengths(stats):
    content_lengths = pd.concat(
        {
            "summaries_nb_chars": stats.loc[stats["suffix"] == ".txt", "nb_chars"].describe(),
            "summaries_nb_tokens": stats.loc[stats["suffix"] == ".txt", "nb_tokens"].describe(),
            "documents_nb_chars": stats.loc[stats["suffix"] != ".txt", "nb_chars"].describe(),
            "documents_nb_tokens": stats.loc[stats["suffix"] != ".txt", "nb_tokens"].describe()
        },
        axis=1
    )

    return content_lengths

def compute_and_save_nb_occ_article_plot(path, stats):
    stats_without_summaries = stats[stats["suffix"] != ".txt"].copy()
    plt.hist(stats_without_summaries["nb_occ_article"], bins=60)
    plt.xlabel("Nombre d'occurences")
    plt.ylabel("Nombre de documents")
    plt.title("Nombre d’occurrences du pattern 'article' par document")
    plt.savefig(path / "plot_nb_occ_article.png")

__all__ = ["compute_corpus_metadata_stats", "compute_corpus_content_stats", "save_stats", "compute_content_lengths", "compute_and_save_nb_occ_article_plot"]
