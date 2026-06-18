# Embedding

- Cost: Qwen/Qwen3-Embedding-8B using basics pdfs -> 0.0005 * (7851804 / 1000000) = 0.003925902 (less because char and not token)
- Time: 3 minutes for indexing 3207 pages

# Vector database - Qdrant

- Use https://qdrant.tech/documentation/guides/installation/ then `docker compose up -d` from the root of the repository

# Logs - Langfuse

- Use https://langfuse.com/self-hosting/deployment/docker-compose then `docker compose up -d` from the langfuse directory

# Local (dev)

## Init DB

```shell
PYTHONPATH="$PWD/src" python scripts/build_corpus.py
PYTHONPATH="$PWD/src" python scripts/index_corpus.py
PYTHONPATH="$PWD/src" chainlit run app/app.py -w
```

## Run Chatbot

```shell
(cd app && PYTHONPATH="../src" chainlit run app.py -w)
```

# VM (test + prod)

## Deploy with Ansible

`ansible-playbook -i ops/inventory.yml ops/playbook.yml`

## Access

| Environment | Chatbot                               | Langfuse                              | Qdrant                                          |
|-------------|---------------------------------------|---------------------------------------|-------------------------------------------------|
| Test        | https://polylex-chatbot-test.epfl.ch/ | http://itswbhst0031.xaas.epfl.ch:3000 | http://itswbhst0031.xaas.epfl.ch:6333/dashboard |
| Prod        | https://polylex-chatbot.epfl.ch/      | http://itswbhst0030.xaas.epfl.ch:3000 | http://itswbhst0030.xaas.epfl.ch:6333/dashboard |
