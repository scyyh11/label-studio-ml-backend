<!--
---
title: Transcribe text from images with PaddleOCR v5
type: guide
tier: all
order: 40
hide_menu: true
hide_frontmatter_title: true
meta_title: PaddleOCR v5 model connection for transcribing text in images
meta_description: The PaddleOCR v5 model connection integrates the capabilities of PP-OCRv5 with Label Studio to assist in machine learning labeling tasks involving Optical Character Recognition (OCR).
categories:
    - Computer Vision
    - Optical Character Recognition
    - PaddleOCR
image: "/guide/ml_tutorials/paddleocr.png"
---
-->

# PaddleOCR v5 Model Connection

The [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) model connection is a powerful tool that integrates the capabilities of PP-OCRv5 with Label Studio. It is designed to assist in machine learning labeling tasks, specifically those involving Optical Character Recognition (OCR).

## Features

- **PP-OCRv5 Support**: Uses the latest PP-OCRv5 models with 13% improvement over PP-OCRv4
- **Multiple Model Types**: Choose between server (high accuracy) or mobile (lightweight) models
- **Multi-language Support**: Chinese, English, Japanese, Korean, Arabic, Cyrillic, Devanagari, and Latin
- **Optional Preprocessing**: Document orientation classification, document unwarping, and text line orientation detection
- **Docker Ready**: Easy deployment with Docker and docker-compose
- **GPU Support**: Optional GPU inference with PaddlePaddle-GPU

## Model Variants

### PP-OCRv5 Server
- **Detection**: PP-OCRv5_server_det (83.8% Hmean)
- **Recognition**: PP-OCRv5_server_rec (86.38% accuracy)
- Best for: High-accuracy server deployments

### PP-OCRv5 Mobile
- **Detection**: PP-OCRv5_mobile_det (79.0% Hmean)
- **Recognition**: PP-OCRv5_mobile_rec (81.29% accuracy)
- Best for: Edge devices and resource-constrained environments

## Before You Begin

Before you begin, you must install the [Label Studio ML backend](https://github.com/HumanSignal/label-studio-ml-backend?tab=readme-ov-file#quickstart).

This tutorial uses the [`paddleocr` example](https://github.com/HumanSignal/label-studio-ml-backend/tree/master/label_studio_ml/examples/paddleocr).

## Labeling Configuration

Use this labeling configuration in your Label Studio project:

```xml
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
```

## Running with Docker (Recommended)

1. Start the Machine Learning backend on `http://localhost:9090`:

```bash
docker-compose up
```

2. Validate that backend is running:

```bash
curl http://localhost:9090/health
{"status":"UP"}
```

3. Create a project in Label Studio. Then from the **Model** page in the project settings, [connect the model](https://labelstud.io/guide/ml#Connect-the-model-to-Label-Studio). The default URL is `http://localhost:9090`.

## Running with Label Studio (Complete Setup)

To run both the ML backend and Label Studio together:

```bash
docker-compose --profile with-label-studio up
```

This will start:
- PaddleOCR ML backend on `http://localhost:9090`
- Label Studio on `http://localhost:8080`

## Building from Source

To build the ML backend from source:

```bash
docker-compose build
```

## Running without Docker

1. Clone the repository and install dependencies:

```bash
python -m venv ml-backend
source ml-backend/bin/activate
pip install -r requirements-base.txt
pip install -r requirements.txt
```

2. Start the ML backend:

```bash
label-studio-ml start ./paddleocr
```

Or run directly with Python:

```bash
python _wsgi.py
```

## Configuration

### Label Studio Field Mapping

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE_DATA_KEY` | `image` | Key in task data containing the image |
| `IMAGE_TO_NAME` | `image` | Name of the `<Image>` tag |
| `BBOX_FROM_NAME` | `bbox` | Name of the `<Rectangle>` tag |
| `LABEL_FROM_NAME` | `label` | Name of the `<Labels>` tag |
| `TEXT_FROM_NAME` | `transcription` | Name of the `<TextArea>` tag |
| `PREDICTED_LABEL` | `Text` | Label value to apply to detected text |

### PaddleOCR Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `PADDLEOCR_VERSION` | `PP-OCRv5` | OCR version (`PP-OCRv5` or `PP-OCRv4`) |
| `PADDLEOCR_MODEL_TYPE` | `server` | Model type (`server` or `mobile`) |
| `PADDLEOCR_LANG` | `ch` | Language code (see supported languages below) |
| `PADDLEOCR_USE_GPU` | `false` | Enable GPU inference |
| `PADDLEOCR_USE_DOC_ORIENTATION` | `false` | Enable document orientation classification |
| `PADDLEOCR_USE_DOC_UNWARPING` | `false` | Enable document unwarping |
| `PADDLEOCR_USE_TEXTLINE_ORIENTATION` | `true` | Enable text line orientation detection |
| `PADDLEOCR_SCORE_THRESHOLD` | `0.5` | Minimum confidence score (0.0-1.0) |

### Server Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `BASIC_AUTH_USER` | | Basic auth username for the model server |
| `BASIC_AUTH_PASS` | | Basic auth password for the model server |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `WORKERS` | `1` | Number of gunicorn workers |
| `THREADS` | `8` | Number of threads per worker |
| `MODEL_DIR` | `/data/models` | Directory for model storage |
| `LABEL_STUDIO_URL` | | Label Studio instance URL |
| `LABEL_STUDIO_API_KEY` | | Label Studio API key |

### Supported Languages

PP-OCRv5 supports multiple languages:

| Code | Language |
|------|----------|
| `ch` | Chinese (Simplified + Traditional) |
| `en` | English |
| `japan` | Japanese |
| `korean` | Korean |
| `arabic` | Arabic |
| `cyrillic` | Cyrillic languages |
| `devanagari` | Devanagari scripts |
| `latin` | Latin-based languages |

## GPU Support

To enable GPU inference:

1. Install CUDA-enabled PaddlePaddle:
```bash
pip install paddlepaddle-gpu
```

2. Set environment variable:
```bash
export PADDLEOCR_USE_GPU=true
```

3. For Docker, use NVIDIA runtime in `docker-compose.yml`:
```yaml
paddleocr:
  runtime: nvidia
  environment:
    - PADDLEOCR_USE_GPU=true
```

## Troubleshooting

### Images visible in Label Studio but ML backend can't read them

If using local files, ensure the ML backend container can access them:

```yaml
# docker-compose.yml
volumes:
  - /path/to/your/images:/data:ro
```

And set `LABEL_STUDIO_URL` and `LABEL_STUDIO_API_KEY` appropriately.

### ML backend returns empty predictions

1. Check that at least one task is imported in Label Studio
2. Verify the model is connected (Settings > Model)
3. Check backend logs: `docker logs paddleocr`
4. Ensure image paths are accessible from the backend
5. Try lowering `PADDLEOCR_SCORE_THRESHOLD` if confidence filtering is too strict

### Model loading is slow

On first run, models are downloaded automatically. Subsequent runs use cached models. To pre-download:

```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(ocr_version="PP-OCRv5", lang="ch")
```

### Memory issues

If you encounter memory issues:
- Reduce `WORKERS` to 1
- Use the mobile model type: `PADDLEOCR_MODEL_TYPE=mobile`
- Consider using GPU for faster processing

## API Reference

The ML backend exposes the standard Label Studio ML backend API:

- `POST /predict` - Get predictions for tasks
- `POST /setup` - Configure the model
- `GET /health` - Health check endpoint

## License

This project is licensed under the Apache License 2.0.

## References

- [PaddleOCR Documentation](https://paddlepaddle.github.io/PaddleOCR/)
- [Label Studio ML Backend Documentation](https://labelstud.io/guide/ml.html)
- [PP-OCRv5 Technical Report](https://github.com/PaddlePaddle/PaddleOCR)
