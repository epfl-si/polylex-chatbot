# Comparaison des configurations (4 juillet 2026)

Afin de simplifier la recherche du meilleur système de RAG, il a été décidé de séparer l'optimisation hyperparamètres liés à la récupération de ceux liés à la génération.

Cette décision peut être justifiée par le fait que l'évaluation de la partie génération est fortement influencée par la construction du contexte fourni au LLM.
Ce contexte dépend bien évidemment des chunks obtenus lors de l’étape de récupération, mais également de la manière dont ces chunks sont "transformés" avant d’être transmis au modèle (utilisation des chunks bruts, filtre sur les chunks, ajout d'autres chunks, transmission du contenu des documents au complet, ...).
Ce dernier aspect constitue un hyperparamètre à part entière, qui semble plus pertinent à optimiser une fois qu'une configuration optimale a été trouvée pour la partie récupération du système de RAG.

L'optimisation des différents hyperparamètres s'est faite en se basant sur les résultats obtenus sur le jeu de questions de développement.

## Description des hyperparamètres fixés

### Récupération

- corpus + filtres pour écarter certains documents
- loader / parser
- méthode pour nettoyer le contenu textuel
- splitter
- contextualisation des chunks
- BM25 model
- embedder
- retrieval mode + nb_retrieved
- reranker + nb_reranked

### Génération

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
- *configuration_g* : gte + small chunks (no overlap) -> 9'098 points (PYTHONPATH="$PWD/src" python scripts/index_corpus.py --env-path="./envs/.env.configuration_g" --collection-name="configuration_g" --collection-description="Corpus: 20260702_configurations_comparison_corpus / Chunking: RCTS avec 700 caractères, pas d'overlap et séparation avec le pattern ARTICLE / Contextualisation: ajout du titre / Embedding: Alibaba-NLP/gte-multilingual-base")
- *configuration_h* : gte + medium chunks (no overlap) -> TODO points (PYTHONPATH="$PWD/src" python scripts/index_corpus.py --env-path="./envs/.env.configuration_h" --collection-name="configuration_h" --collection-description="Corpus: 20260702_configurations_comparison_corpus / Chunking: RCTS avec 2000 caractères, pas d'overlap et séparation avec le pattern ARTICLE / Contextualisation: ajout du titre / Embedding: Alibaba-NLP/gte-multilingual-base")
- *configuration_i* : gte + large chunks (no overlap) -> TODO points (PYTHONPATH="$PWD/src" python scripts/index_corpus.py --env-path="./envs/.env.configuration_i" --collection-name="configuration_i" --collection-description="Corpus: 20260702_configurations_comparison_corpus / Chunking: RCTS avec 5000 caractères, pas d'overlap et séparation avec le pattern ARTICLE / Contextualisation: ajout du titre / Embedding: Alibaba-NLP/gte-multilingual-base")

## Recherche de la meilleure configuration pour la partie récupération

L’ordre et la manière dont les différents hyperparamètres ont été fixés résultent d’une réflexion préalable.
Cependant, ces choix ne sont pas nécessairement les plus pertinents, en raison d’un manque de recul et de connaissances approfondies sur l’impact réel de chaque hyperparamètre.

TODO : parler des autres hyperparamètres fixés avant

Les métriques considérées pour comparer les configurations sont le recall@k et le MRR.

Les deux premiers hyperparamètres optimimisés sont la taille des chunks et le modèle d'embeddings.

Pour ce faire, sept collections ont été créées et évaluées sur le jeu de développement en comparant les résultats sur une recherche sparse et une recherche dense.

L'analyse des résultats a permis d'identifier deux collections optimales :
- chunks de 2'000 caractères embeddés avec [bge](https://huggingface.co/BAAI/bge-m3)
- chunks de 2'000 caractères embeddés avec [qwen](https://huggingface.co/Qwen/Qwen3-Embedding-8B).

Les deux hyperparamètres suivants qui ont été fixés sont le modèle de reranking et le mode de récupération, en se basant uniquement sur les performances obtenues sur les deux collections mentionnées ci-dessus.

Deux modèles de reranking ont été comparés sur la recherche sparse, dense et hybride :
- [bge](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [qwen](https://huggingface.co/Qwen/Qwen3-Reranker-8B).

Il s'avère que dans tous les cas le reranking améliore les performances, cependant le modèle de reranking bge classe les chunks pertinents bien plus haut dans le classement que le modèle de reranking qwen, c'est donc le reranker bge qui a été sélectionné pour la suite de l'optimisation du système de RAG.

Il est à noter que les performances de la recherche dense sont relativement basses pour toutes les configurations testées, alors que c'est l'inverse pour la recherche sparse. Dans ce contexte, utiliser un mode de récupération hybride peut ne pas être efficace.

En comparant les résultats obtenus entre les configurations "sparse + reranker" et "hybrid + reranker", on constate que la première configuration obtient de meilleures performances sur le recall en considérant k < 5. Cependant, en considérant k = 5 et k = 20, les performances sont similaires entre la recherche sparse ou hybride. Il a tout de même été décidé de considérer la recherche hybride comme hyperparamètre optimal.

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

### Évaluation de l'utilisation d'un reranker et de la recherche hybride

#### Analyse de la pertinence du reranker sur les résultats sparse, dense et hybrid

On veut s'assurer que le reranking améliore séparément les performances pour la recherche dense et la recherche hybride.

Deux rerankers ont été comparés:
- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [Qwen/Qwen3-Reranker-8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B).

Pour la recherche dense, les deux rerankers améliorent les scores obtenus pour les métriques recall@k sur les configurations B et E.

Pour la recherche hybride, il est intéressant de constater que le reranker BGE classe les chunks pertinents bien plus haut dans le classement que le reranker Qwen sur les configurations B et E.

Le reranker BGE est donc le modèle sélectionné pour la suite des comparaisons.

#### Résultats

Dense + no reranker: voir ci-dessus

Sparse + no reranker: voir ci-dessus

Hybrid + no reranker:
- *configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T12:51:09.372656Z* -> hit_1=0.2308, hit_20=0.6154 et mrr=0.3846 (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_b" --run-description="CONFIGURATION COMPARISONS - configuration_b (only retrieval compared : hybrid + no reranker)")
- *configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T12:49:47.823429Z* -> hit_1=0.1538, hit_20=0.6923 et mrr=0.3564 (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_e" --run-description="CONFIGURATION COMPARISONS - configuration_e (only retrieval compared : hybrid + no reranker)")

Dense + reranker:
- *configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T05:29:00.559288Z* -> hit_1=0.0769, hit_20=0.2308 et mrr=0.0974 (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_b" --run-description="CONFIGURATION COMPARISONS - configuration_b (only retrieval compared : dense + reranker (bge))")
- *configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T05:29:06.867785Z* -> hit_1=0.0769, hit_20=0.1538 et mrr=0.0865 (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_e" --run-description="CONFIGURATION COMPARISONS - configuration_e (only retrieval compared : dense + reranker (bge))")
- *configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T05:46:37.467092Z* -> hit_1=0.0769, hit_20=0.2308 et mrr=0.1047 (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_b" --run-description="CONFIGURATION COMPARISONS - configuration_b (only retrieval compared : dense + reranker (qwen))")
- *configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T05:46:52.093932Z* -> hit_1=0.0000, hit_20=0.1538 et mrr=0.0423 (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_e" --run-description="CONFIGURATION COMPARISONS - configuration_e (only retrieval compared : dense + reranker (qwen))")

Sparse + reranker:
- *configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T08:17:24.483281Z* -> hit_1=0.6154, hit_20=0.8462 et mrr=0.7009 (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_b" --run-description="CONFIGURATION COMPARISONS - configuration_b (only retrieval compared : sparse + reranker (bge))")
- *configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T08:17:34.690492Z* -> hit_1=0.6154, hit_20=0.8462 et mrr=0.7009 (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_e" --run-description="CONFIGURATION COMPARISONS - configuration_e (only retrieval compared : sparse + reranker (bge))")

Hybrid + reranker:
- *configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T07:36:38.116337Z* -> hit_1=0.5385, hit_20=0.8462 et mrr=0.6353 (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_b" --run-description="CONFIGURATION COMPARISONS - configuration_b (only retrieval compared : hybrid + reranker (bge))")
- *configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T07:36:43.464579Z* -> hit_1=0.5385, hit_20=0.8462 et mrr=0.6418 (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_e" --run-description="CONFIGURATION COMPARISONS - configuration_e (only retrieval compared : hybrid + reranker (bge))")
- *configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T07:49:53.180516Z* -> hit_1=0.3846, hit_20=0.8462 et mrr=0.5115 (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_b" --run-description="CONFIGURATION COMPARISONS - configuration_b (only retrieval compared : hybrid + reranker (qwen))")
- *configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T07:49:58.135147Z* -> hit_1=0.4615, hit_20=0.8462 et mrr=0.5606 (PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_e" --run-description="CONFIGURATION COMPARISONS - configuration_e (only retrieval compared : hybrid + reranker (qwen))")

#### Commandes pour obtenir les analyses des évaluations

Dense + no reranker:
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_b" --run-name="configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:18:42.556574Z" --name-dir="retrieval_dense_no_reranker"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_e" --run-name="configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:19:38.398217Z" --name-dir="retrieval_dense_no_reranker"

Sparse + no reranker:
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_b" --run-name="configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:40:06.750436Z" --name-dir="retrieval_sparse_no_reranker"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_e" --run-name="configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:40:54.536944Z" --name-dir="retrieval_sparse_no_reranker"

Hybrid + no reranker:
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_b" --run-name="configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T12:51:09.372656Z" --name-dir="retrieval_hybrid_no_reranker"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_e" --run-name="configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T12:49:47.823429Z" --name-dir="retrieval_hybrid_no_reranker"

Dense + reranker:
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_b" --run-name="configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T05:29:00.559288Z" --name-dir="retrieval_dense_reranker_bge"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_b" --run-name="configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T05:46:37.467092Z" --name-dir="retrieval_dense_reranker_qwen"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_e" --run-name="configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T05:29:06.867785Z" --name-dir="retrieval_dense_reranker_bge"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_e" --run-name="configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T05:46:52.093932Z" --name-dir="retrieval_dense_reranker_qwen"

Sparse + reranker:
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_b" --run-name="configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T08:17:24.483281Z" --name-dir="retrieval_sparse_reranker_bge"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_e" --run-name="configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T08:17:34.690492Z" --name-dir="retrieval_sparse_reranker_bge"

Hybrid + reranker:
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_b" --run-name="configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T07:36:38.116337Z" --name-dir="retrieval_hybrid_reranker_bge"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_b" --run-name="configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T07:49:53.180516Z" --name-dir="retrieval_hybrid_reranker_qwen"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_e" --run-name="configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T07:36:43.464579Z" --name-dir="retrieval_hybrid_reranker_bge"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_e" --run-name="configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T07:49:58.135147Z" --name-dir="retrieval_hybrid_reranker_qwen"

## Recherche de la meilleure configuration pour la partie génération

TODO : analyse résumée pour rapport

### Mise en place de l'environnement

1. copie du ./envs/.env.configuration_e dans ./envs/.env.generation
2. réactivation des LLM-as-Judges dans Langfuse et check que communication entre Langfuse et RCP soit ok (TODO)

### Procédure pour créer un run avec une certaine configuration LLM

1. création d'un .env puis :
  - llm -> modification de `MODEL_LLM_NAME` et `MODEL_LLM_API_KEY`
  - prompt -> modification de `PROMPT_TEMPLATE_FR` et `PROMPT_TEMPLATE_EN`
2. `PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="<env_file>" --run-description="<run_description>"`

# TODO : modifier les valeurs dans les 3 .env créés
# TODO : tester a chaque fois sur les 3 LLMs

### Evaluations avec un contexte contenant toujours 5 chunks

#### Résultats

TODO

#### Analyse

TODO

### Evaluations avec un contexte contenant au maximum 5 chunks selon threshold

#### Résultats

TODO

#### Analyse

TODO

### Evaluations avec un contexte contenant deux documents en entier sauf exception de longueur

#### Résultats

TODO

#### Analyse

TODO

### Evaluations avec un contexte contenant des chunks ou des documents

#### Résultats

TODO

#### Analyse

TODO

# TODO : considérer les ttests

# TODO : run la baseline et ttest pour comparer avec configuration optimale
