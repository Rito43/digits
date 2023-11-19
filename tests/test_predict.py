import pytest
from unittest.mock import patch
from api.app import app, predict_digit

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

@patch('api.app.predict_digit')
def test_post_predict(mockpred, client):
    for digit in range(10):
        mockpred.return_value = digit
        response = client.post("/predict", json={"img1": [0.1] * 784, "img2": [0.1] * 784})
        assert response.status_code == 200
        assert response.json['result'] == (digit == digit)