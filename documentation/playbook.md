# Marche à suivre pour exécuter le pipeline

Certaines variables sont écrites dans le fichier d'environnement durant l'exécution des scripts, il est donc important au départ de dupliquer le contenu du *.env* dans un autre fichier d'environnement et référencer ce nouveau fichier lors de l'exécution des différents scripts.

## 1. Création du corpus (à faire une fois)

Les valeurs des variables pour Langfuse doivent être configurées, puis exécuter :
```script
PYTHONPATH="$PWD/src" python scripts/build_corpus.py --corpus-name="<corpus_name>"
```

Le résultat obtenu est le suivant :
- le dossier *./data/<corpus-name>* est créé et les documents à indexer y sont stockés (pdf, docx et txt)
- le dossier *./stats/<corpus-name>* est créé et le fichier json contenant les métadonnées du corpus y est sauvegardé
- la variable *CORPUS_NAME* est ajoutée dans le fichier d'environnement.

## 2. Analyse du corpus (à faire une fois)

Exécuter :
```script
PYTHONPATH="$PWD/src" python scripts/compute_stats.py --env-path="<env_path>"
```

Les résultats se trouvent sous *./stats/<corpus_name>/...*.

## 3. Indexation du corpus (à faire n_configurations fois si on souhaite améliorer le RAG)

Trois hyperparamètres peuvent être ajustés au moment de l'indexation du corpus :
- la stratégie de chunking
- la manière de contextualiser les chunks
- le modèle d'embeddings utilisé.

Les valeurs des variables pour le modèle d'embeddings ainsi que le chunking doivent être configurées.

Une description de la configuration du système doit également être fournie afin d'être stockée dans les métadonnées de la collection qui sera créée dans la base de données vectorielle.

Elle peut par exemple prendre la forme suivante :
```text
Corpus: 20260512_corpus / Chunking: RCTS avec 1'000 caractères, overlap de 300 et séparation avec le pattern ARTICLE / Contextualisation: ajout du titre et de la catégorie / Embedding: BAAI/bge-m3
```

Le script d'indexation peut finalement être exécuté :
```script
PYTHONPATH="$PWD/src" python scripts/index_corpus.py --env-path="<env_path>" --collection-name="<collection_name>" --collection-description="<collection_description>"
```

Le résultat obtenu est le suivant :
- une collection est créée dans Qdrant
- les chunks ainsi que leur distribution sont respectivement sauvegardés dans *./stats/<corpus_name>/<collection_name>/chunks.txt* et *./stats/<corpus_name>/<collection_name>/plot_chunks_distribution.png*
- les variables *AVG_LEN_FR*, *AVG_LEN_EN* et *DB_COLLECTION_NAME* sont ajoutées dans le fichier d'environnement.

## 4. Création de Datasets dans Langfuse (à faire une fois ou lorsque le jeu de données est modifié)

Exécuter :
```script
PYTHONPATH="$PWD/src" python scripts/create_langfuse_datasets.py --env-path="<env_path>" --dataset-path="./questions_dataset.csv" --dev-dataset-name="<dev_dataset_name>" --test-dataset-name="<test_dataset_name>"
```

Aller sur Langfuse (partie Dataset) pour visualiser les datasets créés.

## 5. Lancement d’une évaluation (à faire n_configurations fois)

Les valeurs des variables pour le modèle de reranking, le LLM et les modèles LLM-as-Judge doivent être configurées.

Exécuter :
```script
PYTHONPATH="$PWD/src" python scripts/trigger_run.py --env-path="<env_path>" --run-description="<run_description>"
```

Aller sur Langfuse (partie Experiment) pour visualiser les scores obtenus aux différentes métriques.

## 6. Analyse d'une évaluation (à faire n_configurations fois)

La valeur du paramètre *run-name* correspond à la première colonne (*Name*) dans l'onglet *Experiments* sur la plateforme Langfuse.

Exécuter :
```script
PYTHONPATH="$PWD/src" python scripts/analyze_run.py --env-path="<env_path>" --run-name="<run_name>" --name-dir="<name_output_results_dir"
```

Les résultats se trouvent sous *./evaluations/<corpus_name>/<collection>/<name_output_results_dir>/...*.
