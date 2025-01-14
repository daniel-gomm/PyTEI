# PyTEI
PyTEI is a minimal python interface for Hugging Face's [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference).

PyTEI supports in-memory and persistent caching for text embeddings.

## Installation
First, clone the git repository by running:

```shell
git clone https://github.com/daniel-gomm/PyTEI.git
```

Next, install this repository as python package using pip by running the following command from the [root directory](./) 
of this repository:

```shell
pip install .
```

Add the `-e`-flag in case you want to modify the code.

## Usage
Prerequisite for using PyTEI is a running [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference)
instance, for example a local docker container running TEI. Such a docker contain can be spun-up by running:

```shell
docker run --gpus all -p 8080:80 \
  -v $PWD/data:/data \
  --pull always ghcr.io/huggingface/text-embeddings-inference:1.6 \
  --model-id Alibaba-NLP/gte-Qwen2-1.5B-instruct
```

### TEI Client

> For more details check out the [Documentation](https://daniel-gomm.github.io/PyTEI/).

Establish a connection to TEI through a [TEIClient](./src/pytei/client.py). The client gives you access to the 
text-embedding API of the TEI instance:

```python
from pytei import TEIClient

client = TEIClient(url="127.0.0.1:8080/embed")

text_embedding = client.embed("Lorem Ipsum")
```

The default configuration uses in-memory caching of embeddings. For persistent caching use the 
[DuckDBDataStore](./src/pytei/store.py) or implement your own caching solution by extending the 
[DataStore](./src/pytei/store.py) base-class.

```python
from pytei import TEIClient
from pytei.store import DuckDBEmbeddingStore

persistent_data_store = DuckDBEmbeddingStore(db_path="data/embedding_database.duckdb")
client = TEIClient(embedding_store=persistent_data_store, url="127.0.0.1:8080/embed")

text_embedding = client.embed("Lorem Ipsum")
```

For a more detailed description and the full description of the API check out the 
[Documentation](https://daniel-gomm.github.io/PyTEI/.)
