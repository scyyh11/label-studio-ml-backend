"""
This file contains tests for the API of the PaddleOCR ML backend.

You can run these tests by installing test requirements:

    ```bash
    pip install -r requirements-test.txt
    ```

Then execute `pytest` in the directory of this file.
"""
import os.path
import pytest
import json
from model import PaddleOCR
import responses


@pytest.fixture
def client():
    from label_studio_ml.api import init_app
    app = init_app(model_class=PaddleOCR)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def model_dir_env(tmp_path, monkeypatch):
    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()
    monkeypatch.setattr(PaddleOCR, 'MODEL_DIR', str(model_dir))
    return model_dir


def test_health(client):
    """Test the health endpoint."""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data.get('status') == 'UP'


@responses.activate
def test_predict(client, model_dir_env):
    """Test the predict endpoint with a sample image."""
    # Skip if test image doesn't exist
    test_image_path = os.path.join(os.path.dirname(__file__), 'test_images', 'sample.jpg')
    if not os.path.exists(test_image_path):
        pytest.skip("Test image not found. Create test_images/sample.jpg to run this test.")

    responses.add(
        responses.GET,
        'http://test.paddleocr.ml-backend.com/sample.jpg',
        body=open(test_image_path, 'rb').read(),
        status=200
    )

    request = {
        'tasks': [{
            'id': 1,
            'data': {
                'image': 'http://test.paddleocr.ml-backend.com/sample.jpg'
            }
        }],
        'label_config': '''
<View>
  <Image name="image" value="$image"/>
  <Labels name="label" toName="image">
    <Label value="Text" background="green"/>
    <Label value="Handwriting" background="blue"/>
  </Labels>
  <Rectangle name="bbox" toName="image" strokeWidth="3"/>
  <TextArea name="transcription" toName="image"
            editable="true"
            perRegion="true"
            required="true"
            maxSubmissions="1"
            rows="5"
            placeholder="Recognized Text"
            displayMode="region-list"/>
</View>
'''
    }

    response = client.post('/predict', data=json.dumps(request), content_type='application/json')
    assert response.status_code == 200

    data = json.loads(response.data)
    assert 'results' in data

    # Check that predictions have the expected structure
    if data['results'] and data['results'][0].get('result'):
        for result in data['results'][0]['result']:
            assert 'from_name' in result
            assert 'to_name' in result
            assert 'type' in result
            assert 'value' in result

            # Check specific result types
            if result['type'] == 'rectangle':
                assert 'x' in result['value']
                assert 'y' in result['value']
                assert 'width' in result['value']
                assert 'height' in result['value']
            elif result['type'] == 'labels':
                assert 'labels' in result['value']
            elif result['type'] == 'textarea':
                assert 'text' in result['value']


def test_predict_empty_task(client):
    """Test prediction with an empty task."""
    request = {
        'tasks': [{
            'id': 1,
            'data': {}
        }],
        'label_config': '''
<View>
  <Image name="image" value="$image"/>
  <Labels name="label" toName="image">
    <Label value="Text" background="green"/>
  </Labels>
  <Rectangle name="bbox" toName="image" strokeWidth="3"/>
  <TextArea name="transcription" toName="image" perRegion="true"/>
</View>
'''
    }

    response = client.post('/predict', data=json.dumps(request), content_type='application/json')
    assert response.status_code == 200

    data = json.loads(response.data)
    assert 'results' in data
    # Should return empty results for task with no image
    assert data['results'][0]['result'] == []


def test_setup(client):
    """Test the setup endpoint."""
    request = {
        'project': '1.1234567890',
        'schema': '''
<View>
  <Image name="image" value="$image"/>
  <Labels name="label" toName="image">
    <Label value="Text" background="green"/>
  </Labels>
  <Rectangle name="bbox" toName="image" strokeWidth="3"/>
  <TextArea name="transcription" toName="image" perRegion="true"/>
</View>
''',
        'extra_params': {}
    }

    response = client.post('/setup', data=json.dumps(request), content_type='application/json')
    assert response.status_code == 200

    data = json.loads(response.data)
    assert 'model_version' in data
