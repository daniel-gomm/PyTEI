# PyTEI
PyTEI is an unofficial minimal python interface for Hugging Face's [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference).

PyTEI supports in-memory and persistent caching for text embeddings.

## Installation
Install the package through pip:

```shell
pip install pytei-client
```


### Installing from source

First, clone the git repository by running:

```shell
git clone https://github.com/daniel-gomm/PyTEI.git
```

Next, install this repository as python package using pip by running the following command from the [root directory](./) 
of this repository:

```shell
pip install . -e
```

Remove the `-e`-flag in case you do not want to modify the code.

## Usage
Prerequisite for using PyTEI is a running [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference)
instance, for example a local docker container running TEI. Such a docker contain can be spun-up by running:

```shell
docker run --gpus all -p 8080:80 \
  -v $PWD/data:/data \
  --pull always ghcr.io/huggingface/text-embeddings-inference:1.6 \
  --model-id Alibaba-NLP/gte-large-en-v1.5
```

### TEI Client

> For more details check out the [Documentation](https://daniel-gomm.github.io/PyTEI/).

Establish a connection to TEI through a [TEIClient](./src/pytei/client.py). The client gives you access to the 
text-embedding API of the TEI instance:

```python
from pytei import TEIClient

client = TEIClient(url="127.0.0.1:8080/embed")

text_embedding = client.embed("Lorem Ipsum")

text_embedding_batch = client.embed(["Lorem Ipsum", "dolor sit amet", "consectetur adipiscing elit"])

denormalized_embedding = client.embed("Lorem Ipsum", normalize=False)
```

The default configuration uses in-memory caching of embeddings. For persistent caching use the 
[DuckDBDataStore](./src/pytei/store.py) or implement your own caching solution by extending the 
[DataStore](./src/pytei/store.py) base-class.

```python
from pytei import TEIClient
from pytei.store import DuckDBEmbeddingStore

persistent_data_store = DuckDBEmbeddingStore(db_path="data/embedding_database.duckdb")
client = TEIClient(embedding_store=persistent_data_store, url="127.0.0.1:8080/embed")

text = "Lorem Ipsum"

# Embeddings are cached the first time a given text is embedded
text_embedding = client.embed(text)

# Previously cached embedding is retrieved from cache
cached_text_embedding = client.embed(text)

# You can explicitly specify to skip writing to cache
skipped_cache_embedding = client.embed("This will not be cached", skip_cache=True)

# PyTEI supports retrieval from cache for partly cached batches
embeddings = client.embed([text, "Previously un-cached text"])
```

For a more detailed description and the full description of the API check out the 
[Documentation](https://daniel-gomm.github.io/PyTEI/.)
