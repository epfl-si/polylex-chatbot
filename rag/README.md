# Embedding

- Cost: Qwen/Qwen3-Embedding-8B using basics pdfs -> 0.0005 * (7851804 / 1000000) = 0.003925902 (less because char and not token)
- Time: 3 minutes for indexing 3207 pages

# Vector database - Qdrant

- Use https://qdrant.tech/documentation/guides/installation/ then `docker compose up -d` from the root of the repository

# Logs - Langfuse

- Use https://langfuse.com/self-hosting/deployment/docker-compose then `docker compose up -d` from the langfuse directory

# Init DB

```shell
python -m rag.lib.build_corpus \
  --metadata-dir ./test_stats_dir \
  --data-dir ./test_data_dir
```

```shell
python -m rag.lib.index_corpus \
  --metadata-dir ./test_stats_dir \
  --data-dir ./test_data_dir \
  --chunks-log-path ./test_stats_dir/chunks.txt \
  --collection-name dev_collection
```

# Create docker polylex-chatbot image

`docker build --no-cache -t polylex-chatbot .`

```shell
docker run --env-file ./rag/.env \
  -v ./docker_data:/data \
  polylex-chatbot \
  python -m rag.lib.build_corpus
```

```shell
docker run --env-file ./rag/.env \
  -v ./docker_data:/data \
  polylex-chatbot \
  python -m rag.lib.index_corpus
```
