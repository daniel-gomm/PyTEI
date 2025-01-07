from abc import ABC, abstractmethod
import numpy as np
import duckdb
import pickle

class DataStore(ABC):
    """Abstract interface for a key-value store."""

    @abstractmethod
    def get(self, key: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def put(self, key:str, value: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def remove(self, key:str):
        raise NotImplementedError

class InMemoryDataStore(DataStore):
    """In-memory key-value store for embeddings."""

    def __init__(self):
        self.store = {}

    def get(self, key: str) -> np.ndarray:
        return self.store[key]

    def put(self, key:str, value: np.ndarray):
        self.store[key] = value

    def remove(self, key:str):
        del self.store[key]

class DuckDBDataStore(DataStore):

    def __init__(self, db_path: str = "datastore.duckdb"):
        self.conn = duckdb.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS KeyValueStore (
                key TEXT PRIMARY KEY,
                value BLOB
            )
        """)

    def get(self, key: str) -> np.ndarray:
        result = self.conn.execute("SELECT value FROM KeyValueStore WHERE key = ?", (key,)).fetchone()
        if result is None:
            raise KeyError(f"Key '{key}' not found in the datastore.")
        return pickle.loads(result[0])

    def put(self, key: str, value: np.ndarray):
        serialized_value = pickle.dumps(value)
        self.conn.execute("""
            INSERT INTO KeyValueStore (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """, (key, serialized_value))

    def remove(self, key: str):
        self.conn.execute("DELETE FROM KeyValueStore WHERE key = ?", (key,))