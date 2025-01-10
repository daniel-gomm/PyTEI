from abc import ABC, abstractmethod
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
    def put(self, key:str, value: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def remove(self, key:str):
        raise NotImplementedError

class InMemoryEmbeddingStore(EmbeddingStore):
    """In-memory key-value store for embeddings."""

    def __init__(self):
        self.store = {}

    def get(self, key: str) -> np.ndarray:
        return self.store[key]

    def put(self, key:str, value: np.ndarray):
        self.store[key] = value

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

    def put(self, key: str, value: np.ndarray):
        serialized_value = pickle.dumps(value)
        self._db_connection.execute("""
            INSERT INTO DataStore (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """, (key, serialized_value))

    def remove(self, key: str):
        self._db_connection.execute("DELETE FROM DataStore WHERE key = ?", (key,))