from fastapi.testclient import TestClient
from app import review


def test_review_persistence_and_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(review, 'DERIVED', tmp_path)
    client = TestClient(review.app)
    asset = client.get('/api/assets?limit=1').json()['items'][0]
    url = f'/api/assets/{asset["id"]}/review'
    region = dict(x=0, y=0, width=10, height=10, group='A', confirmed=True)
    assert client.put(url, json={'regions': [region]}).status_code == 422
    region.update(label='reviewed test label', reviewer='test', evidence='test fixture')
    assert client.put(url, json={'regions': [region]}).status_code == 200
    assert client.get(url).json()['regions'][0]['label'] == region['label']
    region['width'] = 99999
    assert client.put(url, json={'regions': [region]}).status_code == 422
    assert client.get('/api/assets/nonexistent/image').status_code == 404
    assert client.get('/').status_code == 200
