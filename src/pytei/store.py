from abc import ABC, abstractmethod
from typing import Dict, Collection, List

import numpy as np
import duckdb
import pickle

class EmbeddingStore(ABC):
    """Abstract interface for a key-value store."""

    @abstractmethod
    def get(self, key: str) -> np.ndarray:
        """Get the embedding associated with the specified key. Raises KeyError if the key is not found.
        :param key: The key to get the embedding for.
        :type key: str
        :return: The embedding associated with the specified key.
        :rtype: `numpy.ndarray`
        """
        raise NotImplementedError

    @abstractmethod
    def get_all(self, keys: Collection[str]) -> Dict[str, np.ndarray]:
        """Get the embeddings associated with a set of keys. Returns a dictionary of all found key-value pairs.
        :param keys: The keys to get embeddings for.
        :type keys: Collection[str]
        :return: The embedding associated with the specified key.
        :rtype: `numpy.ndarray`
        """
        raise NotImplementedError

    @abstractmethod
    def put(self, key:str, value: np.ndarray) -> None:
        """
        Store the embedding associated with the specified key.
        :param key: Identifier of the embedding.
        :type key: str
        :param value: Embedding to store.
        :type value: `numpy.ndarray`
        """
        raise NotImplementedError

    @abstractmethod
    def put_all(self, keys: List[str], values: List[np.ndarray]) -> None:
        """
        Store the embeddings associated with the specified keys.
        :param keys: Identifiers of the embeddings to store.
        :type keys: List[str]
        :param values: Embeddings to store.
        :type values: List[`numpy.ndarray`]
        """
        raise NotImplementedError


    @abstractmethod
    def remove(self, key:str) -> None:
        """
        Remove the embedding associated with the specified key.
        :param key: Identifier of the embedding to remove.
        :type key: str
        """
        raise NotImplementedError

class InMemoryEmbeddingStore(EmbeddingStore):
    """In-memory key-value store for embeddings."""

    def __init__(self):
        self.store = {}

    def get(self, key: str) -> np.ndarray:
        return self.store[key]

    def get_all(self, keys: Collection[str]) -> Dict[str, np.ndarray]:
        return {key: value for key, value in self.store.items() if key in keys}

    def put(self, key:str, value: np.ndarray):
        self.store[key] = value

    def put_all(self, keys: List[str], values: List[np.ndarray]) -> None:
        for key, value in zip(keys, values):
            self.put(key, value)

    def remove(self, key:str):
        del self.store[key]

class DuckDBEmbeddingStore(EmbeddingStore):
    """Persistent key-value store using DuckDB as backend."""

    def __init__(self, db_path: str = "datastore.duckdb"):
        """
        :param db_path: Path to the database file. If database does not exist, it will be created.
        :type db_path: str
        """
        self._db_connection = duckdb.connect(db_path)
        self._db_connection.execute("""
            CREATE TABLE IF NOT EXISTS DataStore (
                key TEXT PRIMARY KEY,
                value BLOB
            )
        """)

    def get(self, key: str) -> np.ndarray:
        result = self._db_connection.execute("SELECT value FROM DataStore WHERE key = ?", (key,)).fetchone()
        if result is None:
            raise KeyError(f"Key '{key}' not found in the datastore.")
        return pickle.loads(result[0])

    def get_all(self, keys: Collection[str]) -> Dict[str, np.ndarray]:
        if not keys:
            return {}
        placeholders = ",".join(["?"] * len(keys))
        query = f"SELECT key, value FROM DataStore WHERE key IN ({placeholders})"
        results = self._db_connection.execute(query, keys).fetchall()
        return {key: pickle.loads(value) for key, value in results}

    def put(self, key: str, value: np.ndarray):
        serialized_value = pickle.dumps(value)
        self._db_connection.execute("""
            INSERT INTO DataStore (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """, (key, serialized_value))

    def put_all(self, keys: List[str], values: List[np.ndarray]) -> None:
        serialized_data = [(key, pickle.dumps(value)) for key, value in zip(keys, values)]
        self._db_connection.executemany("""
                INSERT INTO DataStore (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, serialized_data)

    def remove(self, key: str):
        self._db_connection.execute("DELETE FROM DataStore WHERE key = ?", (key,))