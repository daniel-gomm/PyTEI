import requests
import numpy as np
from hashlib import sha1
import json

from pytei.store import DataStore, InMemoryDataStore


class TEIEmbedder:
    """
    A minimal interface for Text Embedding Inference.

    This class communicates with the text embedding endpoint and caches embeddings
    using a specified datastore.
    """

    def __init__(self, data_store: DataStore = None, endpoint: str = "127.0.0.1:8080/embed", timeout: int = 10):
        """
        Args:
            data_store (DataStore): Datastore to cache embeddings.
            endpoint (str): URL of the embedding service.
            timeout (int): Timeout for HTTP requests.
        """
        self.data_store = data_store or InMemoryDataStore()
        self.endpoint = endpoint
        self.timeout = timeout

    def _fetch_embedding(self, text: str) -> np.ndarray:
        """Send a request to the embedding endpoint."""
        try:
            response = requests.post(self.endpoint, json={"inputs": text}, headers={"Content-Type": "application/json"},
                                     timeout=self.timeout)
            response.raise_for_status()  # Raise an HTTPError for non-200 responses
            embedding = json.loads(response.text)[0]  # Expect a single embedding in the response
            return np.array(embedding, dtype=np.float32)
        except (requests.RequestException, json.JSONDecodeError, IndexError, ValueError) as e:
            raise RuntimeError(f"Failed to fetch embedding: {e}")

    def embed(self, text: str) -> np.ndarray:
        """
        Get the embedding for a given text.

        If the embedding is already cached in the datastore, it is retrieved from there.
        Otherwise, it is fetched from the embedding service and cached.

        Args:
            text (str): The text to embed.

        Returns:
            np.ndarray: The embedding as a NumPy array.
        """
        text_hash = sha1(text.encode()).hexdigest()
        try:
            return self.data_store.get(text_hash)
        except KeyError:
            embedding = self._fetch_embedding(text)
            self.data_store.put(text_hash, embedding)
            return embedding
