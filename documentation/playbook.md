# Exécution du workflow complet (2 juillet 2026)

## Création du corpus (à faire une fois)

1. création du .env depuis .env.sample et ajout des valeurs pour Langfuse
2. `PYTHONPATH="$PWD/src" python scripts/build_corpus.py --corpus-name="20260702_configurations_comparison_corpus"`

-> 322 éléments dans le fichier de métadonnées

-> nom du corpus: "20260702_configurations_comparison_corpus"

Le résultat obtenu est le suivant :
- le dossier *./data/<corpus-name>* est créé et les documents à indexer y sont stockés (docx, txt et pdf)
- le dossier *./stats/<corpus-name>* est créé et le fichier json contenant les métadonnées du corpus y est sauvegardé
- la variable *CORPUS_NAME* est ajoutée dans le fichier d'environnement.

## Analyse du corpus (à faire une fois)

3. création du .env.20260702_test_baseline_but_qwen depuis .env
4. `PYTHONPATH="$PWD/src" python scripts/compute_stats.py --env-path=".env.20260702_test_baseline_but_qwen"`

-> résultats sous : ./stats/20260702_configurations_comparison_corpus/...

## Indexation du corpus (à faire n_configurations fois)

Trois hyperparamètres peuvent être ajustés au moment de l'indexation du corpus :
- la stratégie de chunking (1)
- la manière de contextualiser les chunks (2)
- le modèle d'embeddings utilisé (3).

Il est avant tout nécessaire de copier le fichier d'environnement actuel dans un autre fichier (nommé par exemple *.env.<collection-name>*) afin de ne pas écraser la valeur des variables *AVG_LEN_FR*, *AVG_LEN_EN* et *DB_COLLECTION_NAME*.
Les valeurs des variables pour le modèle d'embeddings doivent être configurées.

Une description de la collection doit également être fournie afin d'être stockée dans les métadonnées de la collection dans la base de données.
Elle peut par exemple prendre la forme suivante :
```text
Corpus: 20260701_configs_comparison_corpus / Chunking: RCTS avec 2'000 caractères, overlap de 200 et séparation avec le pattern ARTICLE / Contextualisation: ajout du titre / Embedding: BAAI/bge-m3
```

Le script d'indexation peut finalement être exécuté avec la commande `PYTHONPATH="$PWD/src" python scripts/index_corpus.py --env-path="<env-path>" --collection-name="<collection-name>" --collection-description="<collection-description>"` afin d'indexer le corpus en fonction de la configuration choisie.

5. ajout des valeurs pour modèle d'embeddings et chunking dans .env.20260702_test_baseline_but_qwen
6. `PYTHONPATH="$PWD/src" python scripts/index_corpus.py --env-path=".env.20260702_test_baseline_but_qwen" --collection-name="20260702_test_baseline_but_qwen" --collection-description="Corpus: 20260702_configurations_comparison_corpus / Chunking: RCTS avec 2'000 caractères, overlap de 200 et séparation avec le pattern ARTICLE / Contextualisation: ajout du titre / Embedding: Qwen/Qwen3-Embedding-8B"`

-> collection "20260702_test_baseline_but_qwen" avec 3'070 points

-> chunks sauvegardés dans "./stats/20260702_configurations_comparison_corpus/20260702_test_baseline_but_qwen/chunks.txt"

## Lancement d’une évaluation (à faire n_configurations fois)

- Ajout des valeurs pour reranker, llm et juges dans le fichier d'environnement actuel
- Relancer Langfuse et s'assurer qu'il puisse se connecter à RCP (pour résoudre le problème entre le VPN et Docker)
6. `PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path=".env.20260702_test_baseline_but_qwen"`

-> voir dans Langfuse le dataset run créé

## Analyse d'une évaluation (à faire une fois)

7. `PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path=".env.20260702_test_baseline_but_qwen" --run-name="20260702_test_baseline_but_qwen_mistralai/Mistral-Small-3.2-24B-Instruct-2506 - 2026-07-02T16:09:43.339683Z"`

-> résultats sous : ./evaluations/20260702_configurations_comparison_corpus/20260702_test_baseline_but_qwen/Mistral-Small-3.2-24B-Instruct-2506/...

## Comparaison des configurations (à faire plusieurs fois)

- création de .env.other_llm_20260702_test_baseline_but_qwen depuis .env.20260702_test_baseline_but_qwen et modification du llm
- `PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path=".env.other_llm_20260702_test_baseline_but_qwen"`
- `PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path=".env.other_llm_20260702_test_baseline_but_qwen" --run-name="20260702_test_baseline_but_qwen_swiss-ai/Apertus-8B-Instruct-2509 - 2026-07-02T16:44:29.391732Z"`
