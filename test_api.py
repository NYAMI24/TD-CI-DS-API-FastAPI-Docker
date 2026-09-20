"""
Tests d'intégration de l'API FastAPI — Exercice 1 (TP2).

On utilise TestClient, le client de test fourni par FastAPI : il exécute
l'application en mémoire, sans lancer de serveur ni passer par le réseau.
C'est la façon standard et la plus simple de tester une API FastAPI.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# 1) Prédiction CORRECTE avec [1.0, 2.0, 3.0]
def test_predict_success():
    response = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    # Le modèle apprend y = 2*x, donc on attend [2.0, 4.0, 6.0].
    # pytest.approx tolère les micro-écarts de calcul en virgule flottante.
    assert predictions == pytest.approx([2.0, 4.0, 6.0])


# 2) Prédiction INCORRECTE : la sortie ne doit PAS correspondre à une valeur fausse
def test_predict_wrong_expectation():
    response = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    # [5.0, 5.0, 5.0] est un résultat volontairement faux.
    # Le test réussit précisément parce que l'API renvoie autre chose.
    assert predictions != [5.0, 5.0, 5.0]


# 3) JSON INCORRECT : champ "features" manquant -> FastAPI répond 422
def test_predict_invalid_json():
    # Clé erronée ("feature" au lieu de "features") => champ requis absent.
    response = client.post("/predict", json={"feature": [3.5, 1.2, 4.9]})
    assert response.status_code == 422