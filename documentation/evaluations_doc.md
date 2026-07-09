# Comparaison des configurations (4 juillet 2026)

Afin de simplifier la recherche du meilleur système de RAG, il a été décidé de séparer l'optimisation des hyperparamètres liés à la récupération de ceux liés à la génération.

Cette décision peut être justifiée par le fait que l'évaluation de la partie génération est fortement influencée par la construction du contexte fourni au LLM.
Ce contexte dépend bien évidemment des chunks obtenus lors de l’étape de récupération, mais également de la manière dont ces chunks sont "transformés" avant d’être transmis au modèle (utilisation des chunks bruts, filtre sur les chunks selon leur score, ajout d'autres chunks, transmission du contenu des documents au complet, ...).
Ce dernier aspect constitue un hyperparamètre à part entière, qui semble plus pertinent à optimiser une fois qu'une configuration optimale a été trouvée pour la partie récupération du système de RAG.

L'optimisation des différents hyperparamètres s'est faite en se basant sur les résultats obtenus sur le jeu de questions de développement.

## Description des hyperparamètres fixés

### Récupération

- corpus + filtres pour explure certains documents
- loader (parsing du contenu textuel des documents)
- méthode pour nettoyer le contenu textuel
- splitter (pour créer des chunks)
- méthode pour contextualiser les chunks
- modèle sparse (BM25)
- modèle dense (embeddings)
- mode de récupération et nombre de chunks récupérés
- modèle de reranking et nombre de chunks reclassés

### Génération

- méthode pour construire le contexte envoyé au LLM
- strucutre du prompt
- LLM + température
- seuil de pertinence pour exclure ou non un chunk / document

## Mise en place de l'environnement

1. création du .env depuis .env.sample et ajout des valeurs pour Langfuse
2. utilisation du corpus "20260702_configurations_comparison_corpus" avec 322 documents
3. création du dossier envs et déplacement du .env dedans

## Création des collections

### Procédure pour créer une collection

1. création d'un .env destiné à cette collection puis :
  - chunking -> modification de `CHUNK_SIZE_NB_CHARS` et `CHUNK_OVERLAP_NB_CHARS`
  - embedding -> modification de `MODEL_EMBEDDINGS_NAME`, `MODEL_EMBEDDINGS_API_KEY` et `MODEL_EMBEDDINGS_DIM_VECTOR`
2. `PYTHONPATH="$PWD/src" python scripts/index_corpus.py --env-path="<env_path>" --collection-name="<collection_name>" --collection-description="<collection_description>"`

### Collections créées

| Nom de la collection | Configuration testée (modèle dense / taille des chunks en caractères) | Nombre de chunks créés | Commande exécutée                                                                                                                                                                                                                                                                                                                                                                    |
|----------------------|-----------------------------------------------------------------------|------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| configuration_a      | bge / 700                                                             | 9'098                  | `PYTHONPATH="$PWD/src" python scripts/index_corpus.py --env-path="./envs/.env.configuration_a" --collection-name="configuration_a" --collection-description="Corpus: 20260702_configurations_comparison_corpus / Chunking: RCTS avec 700 caractères, pas d'overlap et séparation avec le pattern ARTICLE / Contextualisation: ajout du titre / Embedding: BAAI/bge-m3"`              |
| configuration_b      | bge / 2'000                                                           | 3'506                  | `PYTHONPATH="$PWD/src" python scripts/index_corpus.py --env-path="./envs/.env.configuration_b" --collection-name="configuration_b" --collection-description="Corpus: 20260702_configurations_comparison_corpus / Chunking: RCTS avec 2000 caractères, pas d'overlap et séparation avec le pattern ARTICLE / Contextualisation: ajout du titre / Embedding: BAAI/bge-m3"`             |
| configuration_c      | bge / 5'000                                                           | 1'546                  | `PYTHONPATH="$PWD/src" python scripts/index_corpus.py --env-path="./envs/.env.configuration_c" --collection-name="configuration_c" --collection-description="Corpus: 20260702_configurations_comparison_corpus / Chunking: RCTS avec 5000 caractères, pas d'overlap et séparation avec le pattern ARTICLE / Contextualisation: ajout du titre / Embedding: BAAI/bge-m3"`             |
| configuration_d      | qwen / 700                                                            | 9'098                  | `PYTHONPATH="$PWD/src" python scripts/index_corpus.py --env-path="./envs/.env.configuration_d" --collection-name="configuration_d" --collection-description="Corpus: 20260702_configurations_comparison_corpus / Chunking: RCTS avec 700 caractères, pas d'overlap et séparation avec le pattern ARTICLE / Contextualisation: ajout du titre / Embedding: Qwen/Qwen3-Embedding-8B"`  |
| configuration_e      | qwen / 2'000                                                          | 3'506                  | `PYTHONPATH="$PWD/src" python scripts/index_corpus.py --env-path="./envs/.env.configuration_e" --collection-name="configuration_e" --collection-description="Corpus: 20260702_configurations_comparison_corpus / Chunking: RCTS avec 2000 caractères, pas d'overlap et séparation avec le pattern ARTICLE / Contextualisation: ajout du titre / Embedding: Qwen/Qwen3-Embedding-8B"` |
| configuration_f      | qwen / 5'000                                                          | 1'546                  | `PYTHONPATH="$PWD/src" python scripts/index_corpus.py --env-path="./envs/.env.configuration_f" --collection-name="configuration_f" --collection-description="Corpus: 20260702_configurations_comparison_corpus / Chunking: RCTS avec 5000 caractères, pas d'overlap et séparation avec le pattern ARTICLE / Contextualisation: ajout du titre / Embedding: Qwen/Qwen3-Embedding-8B"` |

## Recherche de la meilleure configuration pour la partie récupération

### Résumé

L’ordre des expérimentations, la méthode de sélection ainsi que les valeurs retenues pour les différents hyperparamètres résultent d’une réflexion préalable.
Tester les hyperparamètres dans un autre ordre ou explorer d’autres valeurs pourrait bien évidemment conduire à un système RAG plus performant.

Les métriques considérées pour comparer les configurations sur la partie récupération du RAG sont le *recall@k* et le *MRR*.

Les deux premiers hyperparamètres qui ont été optimimisé sont la taille des chunks et le modèle dense.

Pour ce faire, sept collections ont été créées et les résultats sur une recherche sparse ainsi que dense ont été comparés.

L'analyse des résultats a permis d'identifier deux collections optimales :
- chunks de 2'000 caractères embeddés avec [bge](https://huggingface.co/BAAI/bge-m3)
- chunks de 2'000 caractères embeddés avec [qwen](https://huggingface.co/Qwen/Qwen3-Embedding-8B).

Les deux hyperparamètres suivants qui ont été fixés sont le modèle de reranking et le mode de récupération, en se basant uniquement sur les performances obtenues sur les deux collections mentionnées ci-dessus.

Deux modèles de reranking ont été comparés sur la recherche sparse, dense et hybride :
- [bge](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [qwen](https://huggingface.co/Qwen/Qwen3-Reranker-8B).

Il s'avère que dans tous les cas le reranking améliore les performances, cependant le modèle de reranking *bge* classe les chunks pertinents plus haut dans le classement que le modèle de reranking *qwen*, c'est donc le reranker *bge* qui a été sélectionné pour la suite de l'optimisation du système de RAG.

Il est à noter que les performances de la recherche dense sont relativement basses pour toutes les configurations testées, alors que c'est l'inverse pour la recherche sparse. Dans ce contexte, utiliser un mode de récupération hybride peut ne pas être efficace.

En comparant les résultats obtenus entre les configurations "sparse + reranker" et "hybrid + reranker", on constate que la première configuration obtient de meilleures performances sur le recall en considérant k < 5.
Cependant, en considérant k = 5 et k = 20, les performances sont similaires entre la recherche sparse ou hybride.
Il a tout de même été décidé de considérer la recherche hybride comme hyperparamètre optimal.

## Comparaison des collections et des modes de récupération (dense / sparse)

### Configuration avec mode de récupération dense et sans reranker

*Les pires configurations sont la A et la C (embeddings provenant de bge). Les configurations un peu meilleures (dans l'ordre croissant) sont la F, la B et la D (embeddings provenant de qwen et bge). La meilleure configuration est la E (embeddings provenant de qwen avec de grands chunks).*

| Nom de la collection | recall@1 | recall@20 | MRR      | Nom du run dans Langfuse                                                                    | Commande exécutée                                                                                                                                                                                              |
|----------------------|----------|-----------|----------|---------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| configuration_a      | 0.00     | 0.00      | 0.00     | configuration_a_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:13:58.585693Z | `PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_a" --run-description="CONFIGURATION COMPARISONS - configuration_a (only retrieval compared : dense + no reranker)"` |
| configuration_b      | 0.00     | 0.15      | 0.01     | configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:18:42.556574Z | `PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_b" --run-description="CONFIGURATION COMPARISONS - configuration_b (only retrieval compared : dense + no reranker)"` |
| configuration_c      | 0.00     | 0.00      | 0.00     | configuration_c_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:19:25.599367Z | `PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_c" --run-description="CONFIGURATION COMPARISONS - configuration_c (only retrieval compared : dense + no reranker)"` |
| configuration_d      | 0.00     | 0.15      | 0.01     | configuration_d_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:19:32.555443Z | `PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_d" --run-description="CONFIGURATION COMPARISONS - configuration_d (only retrieval compared : dense + no reranker)"` |
| **configuration_e**  | 0.00     | **0.23**  | **0.02** | configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:19:38.398217Z | `PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_e" --run-description="CONFIGURATION COMPARISONS - configuration_e (only retrieval compared : dense + no reranker)"` |
| configuration_f      | 0.00     | 0.07      | 0.01     | configuration_f_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:19:44.643530Z | `PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_f" --run-description="CONFIGURATION COMPARISONS - configuration_f (only retrieval compared : dense + no reranker)"` |

### Configuration avec mode de récupération sparse et sans reranker

*Seule la taille des chunks est variable lors de cette évaluation, raison pour laquelle les performances sont similaires par paire de configurations.*

*Les pires configurations sont la C et la F (grands chunks). Les configurations un peu meilleures sont la A et la D (petits chunks). Les configurations encore un peu meilleures sont la B et la E (chunks moyens).*

| Nom de la collection | recall@1 | recall@20 | MRR      | Nom du run dans Langfuse                                                                    | Commande exécutée                                                                                                                                                                                               |
|----------------------|----------|-----------|----------|---------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| configuration_a      | **0.46** | **0.69**  | 0.53     | configuration_a_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:39:47.536915Z | `PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_a" --run-description="CONFIGURATION COMPARISONS - configuration_a (only retrieval compared : sparse + no reranker)"` |
| **configuration_b**  | **0.46** | **0.69**  | **0.55** | configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:40:06.750436Z | `PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_b" --run-description="CONFIGURATION COMPARISONS - configuration_b (only retrieval compared : sparse + no reranker)"` |
| configuration_c      | 0.38     | 0.62      | 0.50     | configuration_c_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:40:30.561423Z | `PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_c" --run-description="CONFIGURATION COMPARISONS - configuration_c (only retrieval compared : sparse + no reranker)"` |
| configuration_d      | **0.46** | **0.69**  | 0.53     | configuration_d_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:40:42.129346Z | `PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_d" --run-description="CONFIGURATION COMPARISONS - configuration_d (only retrieval compared : sparse + no reranker)"` |
| **configuration_e**  | **0.46** | **0.69**  | **0.55** | configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:40:54.536944Z | `PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_e" --run-description="CONFIGURATION COMPARISONS - configuration_e (only retrieval compared : sparse + no reranker)"` |
| configuration_f      | 0.38     | 0.62      | 0.50     | configuration_f_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:41:08.368409Z | `PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_f" --run-description="CONFIGURATION COMPARISONS - configuration_f (only retrieval compared : sparse + no reranker)"` |

### Analyse intermédiaire

Le classement des configurations sans le reranker est le suivant : E > B > D > A > C.
Les configurations E et B sont les meilleures et seront donc les configurations candidates pour la suite de la recherche du meilleur système de RAG.

## Évaluation de l'utilisation d'un reranker et de la recherche hybride sur les collections *configuration_b* et *configuration_e*

On veut s'assurer que le reranking améliore séparément les performances pour la recherche dense et la recherche hybride.

Deux rerankers ont été comparés:
- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [Qwen/Qwen3-Reranker-8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B).

Pour la recherche dense, les deux rerankers améliorent les scores obtenus pour les métriques *recall@k*.
Pour la recherche sparse, le reranker *bge* améliore également les résultats obtenus.
Pour la recherche hybride, le reranker *bge* classe néanmoins les chunks pertinents plus haut dans le classement que le reranker *qwen* (visible sur le *MRR*).

### Comparaison des configurations

| Configuration testée (collection / mode de récupération / reranker) | recall@1 | recall@20 | MRR      | Nom du run dans Langfuse                                                                    | Commande exécutée                                                                                                                                                                                                   |
|---------------------------------------------------------------------|----------|-----------|----------|---------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| collection_b / dense / no reranker                                  | 0.00     | 0.15      | 0.01     | (voir ci-dessus)                                                                            | (voir ci-dessus)                                                                                                                                                                                                    |
| collection_e / dense / no reranker                                  | 0.00     | 0.23      | 0.02     | (voir ci-dessus)                                                                            | (voir ci-dessus)                                                                                                                                                                                                    |
| collection_b / dense / bge                                          | 0.08     | 0.23      | 0.10     | configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T05:29:00.559288Z | `PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_b" --run-description="CONFIGURATION COMPARISONS - configuration_b (only retrieval compared : dense + reranker (bge))"`   |
| collection_e / dense / bge                                          | 0.08     | 0.15      | 0.09     | configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T05:29:06.867785Z | `PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_e" --run-description="CONFIGURATION COMPARISONS - configuration_e (only retrieval compared : dense + reranker (bge))"`   |
| collection_b / dense / qwen                                         | 0.08     | 0.23      | 0.10     | configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T05:46:37.467092Z | `PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_b" --run-description="CONFIGURATION COMPARISONS - configuration_b (only retrieval compared : dense + reranker (qwen))"`  |
| collection_e / dense / qwen                                         | 0.00     | 0.15      | 0.04     | configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T05:46:52.093932Z | `PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_e" --run-description="CONFIGURATION COMPARISONS - configuration_e (only retrieval compared : dense + reranker (qwen))"`  |
| collection_b / sparse / no reranker                                 | 0.46     | 0.69      | 0.55     | (voir ci-dessus)                                                                            | (voir ci-dessus)                                                                                                                                                                                                    |
| collection_e / sparse / no reranker                                 | 0.46     | 0.69      | 0.55     | (voir ci-dessus)                                                                            | (voir ci-dessus)                                                                                                                                                                                                    |
| **collection_b / sparse / bge**                                     | **0.62** | 0.84      | **0.70** | configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T08:17:24.483281Z | `PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_b" --run-description="CONFIGURATION COMPARISONS - configuration_b (only retrieval compared : sparse + reranker (bge))"`  |
| **collection_e / sparse / bge**                                     | **0.62** | 0.84      | **0.70** | configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T08:17:34.690492Z | `PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_e" --run-description="CONFIGURATION COMPARISONS - configuration_e (only retrieval compared : sparse + reranker (bge))"`  |
| collection_b / sparse / qwen                                        | -        | -         | -        | -                                                                                           | -                                                                                                                                                                                                                   |
| collection_e / sparse / qwen                                        | -        | -         | -        | -                                                                                           | -                                                                                                                                                                                                                   |
| collection_b / hybrid / no reranker                                 | 0.23     | 0.62      | 0.38     | configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T12:51:09.372656Z | `PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_b" --run-description="CONFIGURATION COMPARISONS - configuration_b (only retrieval compared : hybrid + no reranker)"`     |
| collection_e / hybrid / no reranker                                 | 0.15     | 0.69      | 0.36     | configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T12:49:47.823429Z | `PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_e" --run-description="CONFIGURATION COMPARISONS - configuration_e (only retrieval compared : hybrid + no reranker)"`     |
| **collection_b / hybrid / bge**                                     | 0.54     | **0.85**  | 0.64     | configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T07:36:38.116337Z | `PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_b" --run-description="CONFIGURATION COMPARISONS - configuration_b (only retrieval compared : hybrid + reranker (bge))"`  |
| **collection_e / hybrid / bge**                                     | 0.54     | **0.85**  | 0.64     | configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T07:36:43.464579Z | `PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_e" --run-description="CONFIGURATION COMPARISONS - configuration_e (only retrieval compared : hybrid + reranker (bge))"`  |
| **collection_b / hybrid / qwen**                                    | 0.38     | **0.85**  | 0.51     | configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T07:49:53.180516Z | `PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_b" --run-description="CONFIGURATION COMPARISONS - configuration_b (only retrieval compared : hybrid + reranker (qwen))"` |
| **collection_e / hybrid / qwen**                                    | 0.46     | **0.85**  | 0.56     | configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T07:49:58.135147Z | `PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.configuration_e" --run-description="CONFIGURATION COMPARISONS - configuration_e (only retrieval compared : hybrid + reranker (qwen))"` |

## Commandes pour obtenir les analyses des évaluations

Dense + no reranker:
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_b" --run-name="configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:18:42.556574Z" --name-dir="retrieval_dense_no_reranker"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_e" --run-name="configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:19:38.398217Z" --name-dir="retrieval_dense_no_reranker"

Dense + reranker:
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_b" --run-name="configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T05:29:00.559288Z" --name-dir="retrieval_dense_reranker_bge"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_b" --run-name="configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T05:46:37.467092Z" --name-dir="retrieval_dense_reranker_qwen"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_e" --run-name="configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T05:29:06.867785Z" --name-dir="retrieval_dense_reranker_bge"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_e" --run-name="configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T05:46:52.093932Z" --name-dir="retrieval_dense_reranker_qwen"

Sparse + no reranker:
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_b" --run-name="configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:40:06.750436Z" --name-dir="retrieval_sparse_no_reranker"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_e" --run-name="configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T11:40:54.536944Z" --name-dir="retrieval_sparse_no_reranker"

Sparse + reranker:
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_b" --run-name="configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T08:17:24.483281Z" --name-dir="retrieval_sparse_reranker_bge"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_e" --run-name="configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T08:17:34.690492Z" --name-dir="retrieval_sparse_reranker_bge"

Hybrid + no reranker:
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_b" --run-name="configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T12:51:09.372656Z" --name-dir="retrieval_hybrid_no_reranker"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_e" --run-name="configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-04T12:49:47.823429Z" --name-dir="retrieval_hybrid_no_reranker"

Hybrid + reranker:
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_b" --run-name="configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T07:36:38.116337Z" --name-dir="retrieval_hybrid_reranker_bge"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_b" --run-name="configuration_b_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T07:49:53.180516Z" --name-dir="retrieval_hybrid_reranker_qwen"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_e" --run-name="configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T07:36:43.464579Z" --name-dir="retrieval_hybrid_reranker_bge"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.configuration_e" --run-name="configuration_e_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-05T07:49:58.135147Z" --name-dir="retrieval_hybrid_reranker_qwen"

## Recherche de la meilleure configuration pour la partie génération

### Résumé

Trois LLMs avaient initialement été sélectionnés sur la base de la documentation disponible et des benchmarks consultés :
- [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)
- [mistralai/Mistral-Small-3.2-24B-Instruct-2506](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506)
- [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B).

Cependant, certains de ces modèles étaient rarement chargés sur RCP, ce qui entraînait parfois des temps d’attente importants pour pouvoir ensuite utiliser le modèle.
De plus, une fois chargés, certains modèles présentaient des temps de réponse très longs, voire finissaient par un timeout.
Deux autres modèles ont donc été évalués à leur place, en se basant sur ceux disponibles sur RCP et quasiment toujours chargés.

Au final, seul le modèle Mistral a été conservé. Les deux autres modèles ont été remplacés de la façon suivante :
- Llama par une version développée à l'EPFL -> [EPFLiGHT/Llama-33-70b-Legitron](https://huggingface.co/EPFLiGHT/Llama-33-70b-Legitron) (remplacement en raison de timeouts à répétition)
- Qwen par un modèle plus petit -> [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) (remplacement pour bénéficier d'un modèle qui est constamment accessible sans temps d'attente).

Concernant les prompts, un travail de prompt engineering a été effectué afin d'obtenir une réponse pertinente de la part de chaque LLM comparé.

Pour Mistral, un prompt très simple suffit pour que le modèle ne réponde pas à des questions hors contexte, se base sur le contexte à disposition et répond de façon concise dans la langue de l'utilisateur :
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

Le dernier hyperparamètre à fixer est la construction du contexte à fournir au LLM. Plusieurs configurations ont été comparées :
- fournir un nombre fixe de chunks (2, 5 et 20 chunks)
- fournir au maximum 10 chunks selon un seuil de pertinance fixé à 0.2 pour chaque chunk
- fournir les deux documents complets correspondant aux deux chunks les plus pertinents
- fournir, sur la base de 10 chunks récupérés au départ, les documents correspondant aux chunks d'un même document et les chunks uniques si leur score est plus élevé que la limite de 0.2.

Il s'avère que le LLM fournit les réponses les plus optimales lorsqu'il a à disposition les documents en entier.

### Mise en place de l'environnement

1. copie du ./envs/.env.configuration_e dans ./envs/.env.generation
2. activation des LLM-as-Judges dans Langfuse

### Procédure pour créer un run avec une certaine configuration LLM

1. création d'un .env puis :
  - llm -> modification de `MODEL_LLM_NAME` et `MODEL_LLM_API_KEY` dans .env
  - prompt -> modification de `PROMPT_TEMPLATE_FR` et `PROMPT_TEMPLATE_EN` dans .env
  - context -> modification de `NB_MAX_ITEMS_SENT` et fonction appelée `prepare_llm_context(chunks, scores)` dans config.py
2. `PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="<env_path>" --run-description="<run_description>"`

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

La comparaison entre ces configurations a été effectuée sur une autre collection (même corpus mais ajout dans les métadonnées du nombre de tokens du document parent afin de pouvoir gérer la taille du contexte du LLM).

#### Contexte avec un nombre fixe de chunks

Contexte contenant toujours 2 chunks :
- PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.generation_mistral" --run-description="CONFIGURATION COMPARISONS - (only generation compared : 2 chunks in context)"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.generation_mistral" --run-name="20260706_last_collection_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-06T14:05:39.382080Z" --name-dir="generation_mistral_2_chunks_mistral_judge"

Contexte contenant toujours 5 chunks :
- PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.generation_mistral" --run-description="CONFIGURATION COMPARISONS - (only generation compared : 5 chunks in context)"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.generation_mistral" --run-name="20260706_last_collection_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-06T14:19:45.681424Z" --name-dir="generation_mistral_5_chunks_mistral_judge"

Contexte contenant toujours 20 chunks :
- PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.generation_mistral" --run-description="CONFIGURATION COMPARISONS - (only generation compared : 20 chunks in context)"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.generation_mistral" --run-name="20260706_last_collection_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-06T14:23:27.660590Z" --name-dir="generation_mistral_20_chunks_mistral_judge"

#### Contexte avec un nombre maximal de chunks selon limite sur le score

Avec un seuil fixé à 0.2, seuls quelques chunks sont jugés pertinents à envoyer au LLM, il n'est donc nécessaire de considérer qu'une configuration avec au maximum 10 chunks envoyés au LLM.

Contexte contenant au maximum 10 chunks :
- PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.generation_mistral" --run-description="CONFIGURATION COMPARISONS - (only generation compared : max 10 chunks in context)"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.generation_mistral" --run-name="20260706_last_collection_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-06T14:32:02.850657Z" --name-dir="generation_mistral_max_10_chunks_mistral_judge"

#### Contexte avec un nombre fixe de documents

En considérant l'envoi de 2 documents entiers au LLM, il n'y a que quelques cas (moins de cinq) pour lesquels le contexte du LLM serait dépassé.
Dans ces rares cas, seuls les chunks correspondant sont envoyés au LLM (comme la configuration avec l'envoi de 2 chunks).

En augmentant considérablement le contexte sur lequel le LLM peut s'appuyer, des réponses très pertinentes sont générées.

Contexte contenant 2 documents (ou 2 chunks si contexte surchargé) :
- PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.generation_mistral" --run-description="CONFIGURATION COMPARISONS - (only generation compared : 2 docs in context)"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.generation_mistral" --run-name="20260706_last_collection_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-06T16:15:57.181316Z" --name-dir="generation_mistral_2_docs_mistral_judge"

#### Contexte avec des chunks ou des documents selon la pertinence

Pour cette configuration, on considère au départ 10 chunks et le contexte pour le LLM se construit de la façon suivante, dans la limite de la place disponible :
- si plusieurs chunks proviennent d'un même document et que ce document n'est pas trop long, il est inclu au contexte (si le document est trop long, les chunks respectifs sont inclus au contexte)
- si un chunk provient d'un document référencé uniquement une fois mais que son score est supérieur à la limite, il est inclu au contexte
- (les autres chunks ne sont pas considérés).

Cette configuration semble pertinente sur le jeu de questions de développement et permet de tester les différentes façons de construire le contexte.
Le contexte du LLM n'a jamais été dépassé et un ou plusieurs documents sont envoyés au LLM, ainsi qu'un ou plusieurs chunks pertinents pour chaque question.

- PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.generation_mistral" --run-description="CONFIGURATION COMPARISONS - (only generation compared : modular context)"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.generation_mistral" --run-name="20260706_last_collection_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-06T17:19:12.780266Z" --name-dir="generation_mistral_modular_context_mistral_judge"

Une dernière modification apportée est de mentionner de quelle LEX / DOC le contenu textuel provient, à la place d'indiquer *[Chunk i]*, *[Document i]* ou encore *[Item i]*.

- PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="./envs/.env.generation_mistral" --run-description="CONFIGURATION COMPARISONS - (only generation compared : modular context with lex type + lex number as context for each item)"
- PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="./envs/.env.generation_mistral" --run-name="20260706_last_collection_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-06T18:54:57.009253Z" --name-dir="generation_mistral_modular_context_item_referenced_mistral_judge"
