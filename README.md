# Aletheia Detector

AI-text detection API running on Modal. Evaluates text against multiple detectors and returns verdicts.

## Quick start

Test a single model remotely:
```bash
modal run app.py --model fakespot
```

Deploy:
```bash
modal deploy app.py
```

## API

- `GET /health` - service status and available models
- `GET /models` - list available detectors
- `POST /detect` - run detection
  ```json
  {
    "text": "Text to analyze",
    "models": ["fakespot"] // optional; omit to run all
  }
  ```

## Adding a detector

1. Subclass `BaseDetector` in `detectors.py`
2. Implement `load()` and `predict()`
3. Register it in `DetectorService.load()` inside `app.py`

## Architecture

- `app.py` - Modal app + FastAPI endpoints
- `detectors.py` - detector implementations and shared chunking logic
