# Aletheia

AI-text detection API. Runs locally or on [Modal](https://modal.com).

## Run

Local (needs a CUDA GPU):

```bash
uv sync
uv run app.py
```

Modal:

```bash
modal deploy app.py
```

## API

```
GET  /health   service status, model list
GET  /models   description and token window per model
POST /detect   { "text": "...", "models": ["fakespot"] }
```

`models` is optional; omit to run all.

Score: `P(AI) ∈ [0, 1]`, `>= 0.5` means AI Generated. Text must be 40-5000 whitespace-split words.

## Add a detector

1. Subclass `BaseDetector` in `detectors.py`.
2. Add a `Model(...)` row to `MODELS` in `app.py`.
3. Append the class to `load_trained` in `app.py`.

## Layout

```
app.py          API, engines, Modal services
detectors.py    trained detectors
binoculars.py   Binoculars detector
```
