from typing import Union, List, Literal, Dict, Any, Tuple

import requests
import numpy as np
from hashlib import sha1
import json

from pytei.model import PredictionResult, Rank
from pytei.store import EmbeddingStore, InMemoryEmbeddingStore


class TEIClient:
    """
    A minimal interface for Hugging Face's Text Embeddings Inference.

    This class communicates with the text embedding inference endpoint and caches responses
    using a specified datastore.
    """

    def __init__(self, embedding_store: Union[EmbeddingStore, None] = None, url: str = "http://127.0.0.1:8080", timeout: int = 10):
        """Constructor method

        :param embedding_store: Data store used for cacheing. Defaults to in-memory caching.
        :type embedding_store: class: `pytei.store.DataStore`
        :param url: URL of the TEI service.
        :type url: str
        :param timeout: Timeout in seconds.
        :type timeout: int
        """
        self._data_store = embedding_store or InMemoryEmbeddingStore()
        self._endpoint = url
        self._timeout = timeout

    def _fetch_embedding(self, text: str, body: Dict[str, Any]) -> np.ndarray:
        """Send a request to the embedding endpoint."""
        body["inputs"] = text
        try:
            response = requests.post(f"{self._endpoint}/embed", json=body, headers={"Content-Type": "application/json"},
                                     timeout=self._timeout)
            response.raise_for_status()  # Raise an HTTPError for non-200 responses
            embedding = json.loads(response.text)[0]  # Expect a single embedding in the response
            return np.array(embedding, dtype=np.float32)
        except (requests.RequestException, json.JSONDecodeError, IndexError, ValueError) as e:
            raise RuntimeError(f"Failed to fetch embedding: {e}")

    def _fetch_embeddings(self, texts: List[str], body: Dict[str, Any]) -> List[np.ndarray]:
        """Send a batched request to the embedding endpoint."""
        body["inputs"] = texts
        try:
            response = requests.post(f"{self._endpoint}/embed", json={"inputs": texts}, headers={"Content-Type": "application/json"},
                                     timeout=self._timeout)
            response.raise_for_status()  # Raise an HTTPError for non-200 responses
            embeddings = json.loads(response.text)
            return [np.array(embedding, dtype=np.float32) for embedding in embeddings]
        except (requests.RequestException, json.JSONDecodeError, IndexError, ValueError) as e:
            raise RuntimeError(f"Failed to fetch embedding: {e}")

    @staticmethod
    def _build_embed_call_body(normalize: bool = True, prompt_name: Union[str, None] = None,
                               truncate: bool = False,
                               truncation_direction: Union[Literal['left', 'right'], None] = None) -> Dict[str, Any]:
        body = {"normalize": normalize}
        if prompt_name is not None:
            body["prompt_name"] = prompt_name
        if truncate is True:
            body["truncate"] = True
            if truncation_direction is None:
                truncation_direction = "right"
            body["truncation_direction"] = truncation_direction
        return body


    def embed(self, inputs: Union[str, List[str]], normalize: bool = True, prompt_name: Union[str, None] = None,
              truncate: bool = False, truncation_direction: Union[Literal['left', 'right'], None] = None,
              skip_cache: bool = False) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Get the embedding for a single string or a batch of strings.
        If the embedding is already cached in the datastore, it is retrieved from there.
        Otherwise, it is fetched from the embedding service and cached.

        :param inputs: Single string or list of strings to embed.
        :type inputs: Union[str, List[str]]
        :param normalize: Whether to normalize the embedding. Default is True.
        :type normalize: bool, optional
        :param prompt_name: The name of the prompt that should be used by for encoding. If not set, no prompt will be applied. Must be a key in the 'sentence-transformers' configuration prompts dictionary. Default is None.
        :type prompt_name: str, optional
        :param truncate: Whether to truncate the embedding. Default is False.
        :type truncate: bool, optional
        :param truncation_direction: The direction of the truncation. Default is 'right'.
        :type truncation_direction: str, optional
        :param skip_cache: Whether to skip caching the embedding. Default is False.
        :type skip_cache: bool, optional
        :return: For a single input string the single embedding, for a list of input string a list of corresponding embeddings.
        :rtype: Union[`numpy.ndarray`, List[`numpy.ndarray`]]
        """
        # Create id for call parameters to differentiate distinct parameter combinations in the embedding cache
        call_params_id = ""
        if not normalize:
            call_params_id += "nnormalize"
        if prompt_name is not None:
            call_params_id += f".{prompt_name}"
        if truncate is True:
            call_params_id += f".t.{truncation_direction}"

        if isinstance(inputs, str):
            # Handle embedding single input text
            text_hash = sha1((inputs + call_params_id).encode()).hexdigest()
            try:
                return self._data_store.get(text_hash)
            except KeyError:
                body = self._build_embed_call_body(normalize, prompt_name, truncate, truncation_direction)
                embedding = self._fetch_embedding(inputs, body)
                if not skip_cache:
                    self._data_store.put(text_hash, embedding)
                return embedding
        elif isinstance(inputs, list):
            # Handle embedding a batch of texts
            embedding_results = np.zeros(shape=(len(inputs),), dtype=np.ndarray)
            call_indices = []
            call_texts = []
            text_hashes = [sha1((input_str + call_params_id).encode()).hexdigest() for input_str in inputs]
            cached_embedding_map = self._data_store.get_all(text_hashes)
            for index, input_str in enumerate(inputs):
                try:
                    embedding_results[index] = cached_embedding_map[text_hashes[index]]
                except KeyError:
                    call_indices.append(index)
                    call_texts.append(input_str)
            if len(call_indices) > 0:
                # Only call the embedding endpoint for inputs with cache misses
                body = self._build_embed_call_body(normalize, prompt_name, truncate, truncation_direction)
                embeddings = self._fetch_embeddings(call_texts, body)
                if not skip_cache:
                    keys = [text_hashes[ind] for ind in call_indices]
                    self._data_store.put_all(keys, embeddings)
                embedding_results[call_indices] = embeddings
            return embedding_results.tolist()
        else:
            raise AttributeError("text_input must be either a string or a list of strings.")


    def rerank(self, query: str, texts: List[str], raw_score: bool = False, return_text: bool = False,
               truncate: bool = False, truncation_direction: Union[Literal['left', 'right'], None] = None) -> List[Rank]:
        raise NotImplementedError("Reranking is not yet implemented.")

    def predict(self, inputs: Union[str, Tuple[str, str], List[Union[str, Tuple[str, str]]]], raw_scores: bool = False,
                truncate: bool = False, truncation_direction: Union[Literal['left', 'right'], None] = None) -> Union[PredictionResult, List[PredictionResult]]:
        raise NotImplementedError("Sequence classification is not yet implemented.")