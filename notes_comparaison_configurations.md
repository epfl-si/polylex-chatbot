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

Trois LLMs avaient initialement été sélectionnés sur la base de la documentation disponible et des benchmarks consultés:
- [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)
- [mistralai/Mistral-Small-3.2-24B-Instruct-2506](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506)
- [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B).

Cependant, certains de ces modèles étaient rarement chargés sur RCP, ce qui entraînait parfois des temps d’attente importants pour pouvoir ensuite utiliser le modèle.
De plus, une fois chargés, certains modèles présentaient des temps de réponse très longs, voire finissant par un timeout.

Deux autres modèles ont donc été évalués à leur place, en se basant sur ceux disponibles sur RCP et quasiment toujours chargés.

Au final, seul le modèle Mistral a été conservé. Les deux autres modèles ont été remplacés de la façon suivante :
- Llama par une version développée à l'EPFL -> [EPFLiGHT/Llama-33-70b-Legitron](https://huggingface.co/EPFLiGHT/Llama-33-70b-Legitron) (remplacement en raison de timeouts à répétition)
- Qwen par un modèle plus petit -> [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) (remplacement pour bénéficier d'un modèle qui est constamment accessible sans temps d'attente).

Concernant les prompts, un travail de prompt engineering a été effectué afin d'obtenir une réponse pertinente de la part du LLM.

Pour Mistral, un prompt très simple suffit pour que le modèle ne réponde pas à des questions hors contexte, se base sur le contexte à disposition et réponde de façon concise dans la langue de l'utilisateur :
- PROMPT_TEMPLATE_FR="Réponds à la question en utilisant uniquement le contexte fourni.\n\nContexte:\n{context_text}\n\nQuestion:\n{query}\n\nRéponse:"
- PROMPT_TEMPLATE_EN="Answer the question using only the provided context.\n\nContext:\n{context_text}\n\nQuestion:\n{query}\n\nAnswer:".

Pour les deux autres modèles, ce simple prompt ne suffit absolument pas. En effet, plusieurs comportements non souhaités ont été observés tels que des réponses pour des questions hors contexte (le LLM s'est basé sur ses connaissances internes ou a halluciné) ou des justifications excessives lorsque le LLM se basait sur le contexte fourni pour répondre. Le prompt final pour ces deux modèles est le suivant :
- PROMPT_TEMPLATE_FR="Tu es un assistant juridique chargé de répondre à des questions à partir de documents fournis.\n\nRéponds uniquement à partir du contexte ci-dessous.\nN'utilise aucune connaissance externe.\nSi le contexte ne contient pas l'information nécessaire, réponds exactement : \"Je n'ai pas trouvé d'information pertinente dans les documents fournis.\"\nRéponds directement à la question.\n\nContexte :\n{context_text}\n\nQuestion :\n{query}\n\nRéponse :"
- PROMPT_TEMPLATE_EN="You are a legal assistant tasked with answering questions based on provided documents.\n\nAnswer only using the context below.\nDo not use any external knowledge.\nIf the context does not contain the necessary information, answer exactly: \"I did not find any relevant information in the provided documents.\"\nAnswer the question directly.\n\nContext:\n{context_text}\n\nQuestion:\n{query}\n\nAnswer:".

Malgré ce prompt retravaillé, ces deux LLMs préfèrent parfois répondre qu'ils n'ont pas accès aux informations pertinentes, alors même que l'information est disponible.
Pour cette raison, il est plus judicieux d'utiliser Mistral comme LLM dans le système de RAG et penser à retravailler les prompts avant d'utiliser ces autres modèles.

L'évaluation de la partie génération du système de RAG se base au départ sur des observations humaines, puis sur des évaluations fournies par un LLM-as-Judge.

Dans un premier temps, Mistral a été utilisé comme LLM-as-Judge. Cependant, celui-ci n’était pas toujours disponible sur RCP, Qwen a donc temporairement été utilisé à sa place.
Cette alternative s’est toutefois révélée problématique, car Qwen interprétait parfois incorrectement certaines instructions présentes dans les prompts par défaut des métriques.
Mistral a donc finalement à nouveau été utilisé comme LLM-as-Judge. Ce choix introduit néanmoins un biais potentiel puisque les réponses évaluées proviennent également de Mistral.
C’est pourquoi il a été décidé, en plus d'une réduction du nombre de configurations à tester, de ne pas comparer les différentes stratégies de construction du contexte sur plusieurs LLMs, mais uniquement sur les résultats obtenus avec Mistral.
En effet, une comparaison impliquant plusieurs LLMs évalués par un de ces LLms a de fortes chances d'être biaisée.

TODO : parler des résultats avec les constructions du contexte

### Mise en place de l'environnement

1. copie du ./envs/.env.configuration_e dans ./envs/.env.generation
2. réactivation des LLM-as-Judges dans Langfuse et check que communication entre Langfuse et RCP soit ok

### Procédure pour créer un run avec une certaine configuration LLM

1. création d'un .env puis :
  - llm -> modification de `MODEL_LLM_NAME` et `MODEL_LLM_API_KEY`
  - prompt -> modification de `PROMPT_TEMPLATE_FR` et `PROMPT_TEMPLATE_EN`
2. `PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="<env_file>" --run-description="<run_description>"`

### Evaluations avec un contexte contenant au maximum 5 chunks selon threshold

Mistral semble être le meilleur modèle.
LLama produit des réponses assez brèves, mais répond parfois qu'il n'a pas à disposition l'information pertinente alors que c'est bien le cas.
Qwen agit souvent de la même manière, en plus d'être bien trop verbeux.

#### Commandes

Trigger runs:

- PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.generation_llama" --run-description="CONFIGURATION COMPARISONS - llama (only generation compared : max 5 chunks in context)"
- PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.generation_mistral" --run-description="CONFIGURATION COMPARISONS - mistral (only generation compared : max 5 chunks in context)"
- PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.generation_qwen" --run-description="CONFIGURATION COMPARISONS - qwen (only generation compared : max 5 chunks in context)"
- PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.generation_llama" --run-description="CONFIGURATION COMPARISONS - llama (only generation compared : max 5 chunks in context (with bge as judge))"
- PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.generation_mistral" --run-description="CONFIGURATION COMPARISONS - mistral (only generation compared : max 5 chunks in context (with bge as judge))"
- PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.generation_qwen" --run-description="CONFIGURATION COMPARISONS - qwen (only generation compared : max 5 chunks in context (with bge as judge))"

*note : erreur dans le nom du LLM (bge à la place de mistral)

Analyze runs:

- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.generation_llama" --run-name="configuration_e_EPFLiGHT/Llama-33-70b-Legitron - 2026-07-05T14:01:10.183110Z" --name-dir="generation_llama_max_5_chunks_qwen_judge"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.generation_mistral" --run-name="configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T13:32:43.609971Z" --name-dir="generation_mistral_max_5_chunks_qwen_judge"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.generation_qwen" --run-name="configuration_e_Qwen/Qwen3-30B-A3B-Instruct-2507 - 2026-07-05T13:32:55.257682Z" --name-dir="generation_qwen_max_5_chunks_qwen_judge"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.generation_llama" --run-name="configuration_e_EPFLiGHT/Llama-33-70b-Legitron - 2026-07-05T14:01:10.183110Z" --name-dir="generation_llama_max_5_chunks_mistral_judge"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.generation_mistral" --run-name="configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T14:01:17.395430Z" --name-dir="generation_mistral_max_5_chunks_mistral_judge"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.generation_qwen" --run-name="configuration_e_Qwen/Qwen3-30B-A3B-Instruct-2507 - 2026-07-05T14:01:24.025065Z" --name-dir="generation_qwen_max_5_chunks_mistral_judge"

### Autres configurations pour construire le contexte du LLM

Contexte contenant toujours 5 chunks :
- PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.generation_mistral" --run-description="CONFIGURATION COMPARISONS - mistral (only generation compared : 5 chunks in context (with mistral as judge)"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.generation_mistral" --run-name="configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T16:17:55.556222Z" --name-dir="generation_mistral_5_chunks_mistral_judge"

Contexte contenant 2 documents en entier sauf exception de longueur :
TODO

Contexte contenant 2 documents des chunks ou des documents :
TODO

# TODO : comparer uniquement sur mistral jugé par mistral:
- maximum 5 chunks
- toujours 5 chunks
- 2 documents
- documents ou chunks

# TODO : considérer les ttests

# TODO : run la baseline et ttest pour comparer avec configuration optimale
