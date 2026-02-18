import pytest
from unittest.mock import MagicMock
import mlflow

@pytest.fixture(autouse=True)
def mock_mlflow_model(monkeypatch):
    """
    This 'mocks' MLflow so tests don't need a real mlflow.db file.
    It runs automatically for every test.
    """
    # 1. Create a fake model object
    mock_model = MagicMock()
    
    # 2. Fake the 'predict_proba' method to return a 50/50 probability
    # We mock the internal implementation your main.py uses
    mock_model._model_impl.predict_proba.return_value = [[0.5, 0.5]]
    mock_model._model_impl.sklearn_model = MagicMock() # For SHAP
    
    # 3. Force mlflow.pyfunc.load_model to return our fake model
    monkeypatch.setattr(mlflow.pyfunc, "load_model", lambda *args, **kwargs: mock_model)
    
    # 4. Mock the SHAP explainer to prevent it from crashing without a real model
    monkeypatch.setattr("shap.TreeExplainer", lambda *args, **kwargs: MagicMock())
    
    return mock_model