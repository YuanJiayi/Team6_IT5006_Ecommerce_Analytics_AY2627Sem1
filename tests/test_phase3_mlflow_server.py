"""A store built through the MLflow server is portable and its location check is strict."""

import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

from phase3 import registry  # noqa: E402
from phase3.mlflow_server import mlflow_server  # noqa: E402

IMPORTANCE = "permutation_importance.csv"


def _build_store(store):
    """Log one tiny model by registered name and save an artifact beside it."""
    with mlflow_server(store) as uri:
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)
        client = MlflowClient(tracking_uri=uri, registry_uri=uri)
        experiment = client.create_experiment("tiny")
        with mlflow.start_run(experiment_id=experiment, run_name="tiny"):
            model = LinearRegression().fit([[0.0], [1.0], [2.0]], [0.0, 1.0, 2.0])
            mlflow.sklearn.log_model(model, name="model", registered_model_name="tiny_eta")
            with tempfile.TemporaryDirectory() as tmp:
                artifact = Path(tmp) / IMPORTANCE
                artifact.write_text("input,importance_mean\nx,1.0\n", encoding="utf-8")
                mlflow.log_artifact(str(artifact))
        versions = client.search_model_versions("name = 'tiny_eta'")
        champion = max(versions, key=lambda version: int(version.version))
        client.set_registered_model_alias("tiny_eta", "champion", str(champion.version))


class PortableStoreTests(unittest.TestCase):
    def test_store_location_check_passes_and_a_copy_loads_the_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = root / "store"
            _build_store(store)
            registry.check_store_location(store)

            moved = root / "moved"
            shutil.copytree(store, moved)
            with mlflow_server(moved) as uri:
                mlflow.set_tracking_uri(uri)
                mlflow.set_registry_uri(uri)
                client = MlflowClient(tracking_uri=uri, registry_uri=uri)
                model = mlflow.sklearn.load_model("models:/tiny_eta@champion")
                np.testing.assert_allclose(model.predict([[5.0]]), [5.0])
                version = client.get_model_version_by_alias("tiny_eta", "champion")
                with tempfile.TemporaryDirectory() as download:
                    path = Path(client.download_artifacts(version.run_id, IMPORTANCE, download))
                    self.assertEqual(path.read_text(encoding="utf-8").splitlines()[0],
                                     "input,importance_mean")


class LocationCheckTests(unittest.TestCase):
    def test_a_file_uri_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "store"
            (store / "artifacts").mkdir(parents=True)
            connection = sqlite3.connect(store / "mlflow.db")
            try:
                connection.execute("CREATE TABLE experiments (artifact_location TEXT)")
                connection.execute(
                    "INSERT INTO experiments VALUES (?)",
                    ("file:///Users/someone/artifacts/tiny",))
                connection.commit()
            finally:
                connection.close()
            with self.assertRaises(RuntimeError) as caught:
                registry.check_store_location(store)
            self.assertIn("file:", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
