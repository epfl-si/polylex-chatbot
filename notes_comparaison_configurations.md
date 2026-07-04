# Comparaison de configurations (4 juillet 2026)

## Description des hyperparamètres fixés

### Retrieval

TODO

- corpus + filtres pour écarter certains documents
- loader / parser -> 1
- méthode pour nettoyer le contenu textuel
- splitter -> 6 (RCTS + article pattern + size + overlap + nb chars)
- contextualisation des chunks
- BM25 model
- embedder -> 3 modèles
- retrieval mode -> 3 modes (nb_retrieved=100)
- fusion -> TODO
- reranker -> 1 (nb_reranked=20)

### Generation

TODO

- context -> TODO
- prompt + llm -> 3 modèles avec un prompt propre à chaque modèle
- relevance threshold
- temperature LLM

## Mise en place de l'environnement

1. création du .env depuis .env.sample et ajout des valeurs pour Langfuse
2. utilisation du corpus "20260702_configurations_comparison_corpus" avec 322 documents
3. création du dossier envs et déplacement du .env dedans

## Création des collections

### Procédure pour créer une collection

1. création d'un .env puis :
  - chunking -> modification de `CHUNK_SIZE_NB_CHARS` et `CHUNK_OVERLAP_NB_CHARS`
  - embedding -> modification de `MODEL_EMBEDDINGS_NAME`, `MODEL_EMBEDDINGS_API_KEY` et `MODEL_EMBEDDINGS_DIM_VECTOR`
2. `PYTHONPATH="$PWD/src" python scripts/index_corpus.py --env-path="<env_file>" --collection-name="<collection_name>" --collection-description="<collection_description>"`

### Collections créées

- *configuration_a* : bge + small chunks (no overlap) -> 9'098 points (PYTHONPATH="$PWD/src" python scripts/index_corpus.py --env-path="./envs/.env.configuration_a" --collection-name="configuration_a" --collection-description="Corpus: 20260702_configurations_comparison_corpus / Chunking: RCTS avec 700 caractères, pas d'overlap et séparation avec le pattern ARTICLE / Contextualisation: ajout du titre / Embedding: BAAI/bge-m3")
- *configuration_b* : bge + medium chunks (no overlap) -> 3'506 points (PYTHONPATH="$PWD/src" python scripts/index_corpus.py --env-path="./envs/.env.configuration_b" --collection-name="configuration_b" --collection-description="Corpus: 20260702_configurations_comparison_corpus / Chunking: RCTS avec 2000 caractères, pas d'overlap et séparation avec le pattern ARTICLE / Contextualisation: ajout du titre / Embedding: BAAI/bge-m3")
- *configuration_c* : bge + large chunks (no overlap) -> 1'546 points (PYTHONPATH="$PWD/src" python scripts/index_corpus.py --env-path="./envs/.env.configuration_c" --collection-name="configuration_c" --collection-description="Corpus: 20260702_configurations_comparison_corpus / Chunking: RCTS avec 5000 caractères, pas d'overlap et séparation avec le pattern ARTICLE / Contextualisation: ajout du titre / Embedding: BAAI/bge-m3")
- *configuration_d* : qwen + small chunks (no overlap) -> 9'098 points (PYTHONPATH="$PWD/src" python scripts/index_corpus.py --env-path="./envs/.env.configuration_d" --collection-name="configuration_d" --collection-description="Corpus: 20260702_configurations_comparison_corpus / Chunking: RCTS avec 700 caractères, pas d'overlap et séparation avec le pattern ARTICLE / Contextualisation: ajout du titre / Embedding: Qwen/Qwen3-Embedding-8B")
- *configuration_e* : qwen + medium chunks (no overlap) -> 3'506 points (PYTHONPATH="$PWD/src" python scripts/index_corpus.py --env-path="./envs/.env.configuration_e" --collection-name="configuration_e" --collection-description="Corpus: 20260702_configurations_comparison_corpus / Chunking: RCTS avec 2000 caractères, pas d'overlap et séparation avec le pattern ARTICLE / Contextualisation: ajout du titre / Embedding: Qwen/Qwen3-Embedding-8B")
- *configuration_f* : qwen + large chunks (no overlap) -> 1'546 points (PYTHONPATH="$PWD/src" python scripts/index_corpus.py --env-path="./envs/.env.configuration_f" --collection-name="configuration_f" --collection-description="Corpus: 20260702_configurations_comparison_corpus / Chunking: RCTS avec 5000 caractères, pas d'overlap et séparation avec le pattern ARTICLE / Contextualisation: ajout du titre / Embedding: Qwen/Qwen3-Embedding-8B")
- *configuration_g* : gte + small chunks (no overlap) -> TODO points (PYTHONPATH="$PWD/src" python scripts/index_corpus.py --env-path="./envs/.env.configuration_g" --collection-name="configuration_g" --collection-description="Corpus: 20260702_configurations_comparison_corpus / Chunking: RCTS avec 700 caractères, pas d'overlap et séparation avec le pattern ARTICLE / Contextualisation: ajout du titre / Embedding: Alibaba-NLP/gte-multilingual-base")
- *configuration_h* : gte + medium chunks (no overlap) -> TODO points (PYTHONPATH="$PWD/src" python scripts/index_corpus.py --env-path="./envs/.env.configuration_h" --collection-name="configuration_h" --collection-description="Corpus: 20260702_configurations_comparison_corpus / Chunking: RCTS avec 2000 caractères, pas d'overlap et séparation avec le pattern ARTICLE / Contextualisation: ajout du titre / Embedding: Alibaba-NLP/gte-multilingual-base")
- *configuration_i* : gte + large chunks (no overlap) -> TODO points (PYTHONPATH="$PWD/src" python scripts/index_corpus.py --env-path="./envs/.env.configuration_i" --collection-name="configuration_i" --collection-description="Corpus: 20260702_configurations_comparison_corpus / Chunking: RCTS avec 5000 caractères, pas d'overlap et séparation avec le pattern ARTICLE / Contextualisation: ajout du titre / Embedding: Alibaba-NLP/gte-multilingual-base")

## Evaluation de la partie retrieval

### Procédure pour créer une évaluation

1. modification dans le fichier d'environnement spécifique :
  - reranker
  - llm (configuration baseline car on s'intéresse uniquement au retrieval pour le moment)
  - llm et embedder judges (inactifs pour le moment)
2. `PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="<env_file>" --run-description="<run_description>"`

### Evaluations en se basant sur "dense et sans reranker"

#### Résultats

- *configuration_a_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:13:58.585693Z* -> hit_1=0.0000, hit_20=0.0000 et mrr=0.0000 (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_a" --run-description="CONFIGURATION COMPARISONS - configuration_a (only retrieval compared : dense + no reranker)")
- *configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:18:42.556574Z* -> hit_1=0.0000, hit_20=0.1538 et mrr=0.0147 (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_b" --run-description="CONFIGURATION COMPARISONS - configuration_b (only retrieval compared : dense + no reranker)")
- *configuration_c_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:19:25.599367Z* -> hit_1=0.0000, hit_20=0.0000 et mrr=0.0000 (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_c" --run-description="CONFIGURATION COMPARISONS - configuration_c (only retrieval compared : dense + no reranker)")
- *configuration_d_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:19:32.555443Z* -> hit_1=0.0000, hit_20=0.1538 et mrr=0.0139 (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_d" --run-description="CONFIGURATION COMPARISONS - configuration_d (only retrieval compared : dense + no reranker)")
- *configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:19:38.398217Z* -> hit_1=0.0000, hit_20=0.2308 et mrr=0.0264 (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_e" --run-description="CONFIGURATION COMPARISONS - configuration_e (only retrieval compared : dense + no reranker)")
- *configuration_f_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:19:44.643530Z* -> hit_1=0.0000, hit_20=0.0769 et mrr=0.0154 (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_f" --run-description="CONFIGURATION COMPARISONS - configuration_f (only retrieval compared : dense + no reranker)")
- TODO G -> (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_g" --run-description="CONFIGURATION COMPARISONS - configuration_g (only retrieval compared : dense + no reranker)")
- TODO H -> (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_h" --run-description="CONFIGURATION COMPARISONS - configuration_h (only retrieval compared : dense + no reranker)")
- TODO I -> (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_i" --run-description="CONFIGURATION COMPARISONS - configuration_i (only retrieval compared : dense + no reranker)")

#### Analyse

Les pires configurations sont la A et la C (embeddings provenant de bge-m3).
Les configurations un peu meilleures (dans l'ordre croissant) sont la F, la B et la D (embeddings provenant de qwen et bge-m3).
La meilleure configuration est la E (embeddings provenant de qwen avec de grands chunks).

### Evaluations en se basant sur "sparse et sans reranker"

#### Résultats

- *configuration_a_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:39:47.536915Z* -> hit_1=0.4615, hit_20=0.6923 et mrr=0.5295 (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_a" --run-description="CONFIGURATION COMPARISONS - configuration_a (only retrieval compared : sparse + no reranker)")
- *configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:40:06.750436Z* -> hit_1=0.4615, hit_20=0.6923 et mrr=0.5462 (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_b" --run-description="CONFIGURATION COMPARISONS - configuration_b (only retrieval compared : sparse + no reranker)")
- *configuration_c_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:40:30.561423Z* -> hit_1=0.3846, hit_20=0.6154 et mrr=0.5000 (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_c" --run-description="CONFIGURATION COMPARISONS - configuration_c (only retrieval compared : sparse + no reranker)")
- *configuration_d_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:40:42.129346Z* -> hit_1=0.4615, hit_20=0.6923 et mrr=0.5295 (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_d" --run-description="CONFIGURATION COMPARISONS - configuration_d (only retrieval compared : sparse + no reranker)")
- *configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:40:54.536944Z* -> hit_1=0.4615, hit_20=0.6923 et mrr=0.5462 (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_e" --run-description="CONFIGURATION COMPARISONS - configuration_e (only retrieval compared : sparse + no reranker)")
- *configuration_f_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:41:08.368409Z* -> hit_1=0.3846, hit_20=0.6154 et mrr=0.5000 (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_f" --run-description="CONFIGURATION COMPARISONS - configuration_f (only retrieval compared : sparse + no reranker)")
- TODO G -> (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_g" --run-description="CONFIGURATION COMPARISONS - configuration_g (only retrieval compared : sparse + no reranker)")
- TODO H -> (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_h" --run-description="CONFIGURATION COMPARISONS - configuration_h (only retrieval compared : sparse + no reranker)")
- TODO I -> (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_i" --run-description="CONFIGURATION COMPARISONS - configuration_i (only retrieval compared : sparse + no reranker)")

#### Analyse

Seule la taille des chunks est variable lors de cette évaluation, raison pour laquelle les performances sont similaires par paire de configurations.

Les pires configurations sont la C et la F (grands chunks).
Les configurations un peu meilleures sont la A et la D (petits chunks).
Les configurations encore un peu meilleures sont la B et la E (chunks moyens).

### Comparaison des configurations en se basant sur les configurations DENSE vs SPARSE

Le classement des configurations sans le reranker est le suivant : E > B > D > A > C.
Les configurations E et B sont les meilleures et seront donc les configurations candidates pour la suite de la recherche du meilleur système de RAG.

- A : pire et moyenne sans reranker
- B : moyenne et meilleure sans reranker
- C : pire sans reranker
- D : moyenne sans reranker
- E : meilleure sans reranker
- F : moyenne et pire sans reranker
- G (TODO)
- H (TODO)
- I (TODO)

# TODO : HYBRDID, RRF (weighted ?) et reranker

#### Résultats

- *configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T12:51:09.372656Z* -> hit_1=0.2308, hit_20=0.6154 et mrr=0.3846 (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_b" --run-description="CONFIGURATION COMPARISONS - configuration_b (only retrieval compared : hybrid + no reranker)")
- *configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T12:49:47.823429Z* -> hit_1=0.1538, hit_20=0.6923 et mrr=0.3564 (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_e" --run-description="CONFIGURATION COMPARISONS - configuration_e (only retrieval compared : hybrid + no reranker)")

#### Analyse

Les résultats obtenus avec le mode HYBRID sont moins bons qu'en considérant la configuration "sparse et sans reranker".

## TODO : dense + reranker

#### Résultats

- ** -> hit_1=, hit_20= et mrr= (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_b" --run-description="CONFIGURATION COMPARISONS - configuration_b (only retrieval compared : dense + reranker)")
- ** -> hit_1=, hit_20= et mrr= (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_e" --run-description="CONFIGURATION COMPARISONS - configuration_e (only retrieval compared : dense + reranker)")

# TODO -> est-ce que combinaison hybrid + fusion + reranker mieux que sparse + reranker ?

# TODO : generation

# TODO : comparer avec baseline (battre 0.61 pour recall@1 avec hybrid + reranker)
