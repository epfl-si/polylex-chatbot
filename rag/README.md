# Embedding

- Cost: Qwen/Qwen3-Embedding-8B using basics pdfs -> 0.0005 * (7851804 / 1000000) = 0.003925902 (less because char and not token)
- Time: 3 minutes for indexing 3207 pages

# Vector database - Qdrant

- Use https://qdrant.tech/documentation/guides/installation/ then `docker compose up -d` from the root of the repository
