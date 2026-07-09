# Exemples de requête à exécuter dans Qdrant

## Filtres

```
POST /collections/<collection>/points/scroll
{
    "filter": {
        "must": [
            { "key": "metadata.doc_id", "match": { "value": "<doc_id>" } }
        ]
    }
}
```

## Modes de récupération

### Hybride

```
POST /collections/<collection>/points/query
{
    "prefetch": [
        {
            "query": { 
                "indices": <sparse_query_vector_indices>,
                "values": <sparse_query_vector_values>
             },
            "using": "<name_sparse_vector>",
            "limit": 100
        },
        {
            "query": <dense_query_embedding>,
            "using": "<name_dense_vector>",
            "limit": 100
        }
    ],
    "query": { "fusion": "rrf" },
    "with_payload": true,
    "with_vector": false,
    "limit": 100
}
```

### Dense

```
POST /collections/<collection>/points/query
{
  "query": <dense_query_embedding>,
  "limit": 1000,
  "with_payload": true,
  "with_vector": false
}
```
