import modal
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# App + storage
# ---------------------------------------------------------------------------

app = modal.App("aletheia-detector")
hf_cache = modal.Volume.from_name("aletheia-hf-cache", create_if_missing=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_WORDS = 40
MAX_WORDS = 5000

# Static catalog. Lives at module scope so the (torch-free) web container can
# answer /models and route /detect without booting any GPU service.
MODELS_META = {
    "fakespot": {
        "description": "RoBERTa-base binary AI detector (Fakespot).",
        "window_tokens": 512,
    },
    "szeged": {
        "description": "Three-seed ModernBERT ensemble, 41-class LLM-family classifier (SzegedAI); reports 1 minus P(human).",
        "window_tokens": 2048,
    },
    "desklib": {
        "description": "Mean-pooled transformer with single-logit AI head (Desklib).",
        "window_tokens": 768,
    },
    "superannotate_low_fpr": {
        "description": "RoBERTa-large detector tuned for low false-positive rate (SuperAnnotate).",
        "window_tokens": 512,
    },
    "superannotate": {
        "description": "RoBERTa-large balanced AI detector (SuperAnnotate).",
        "window_tokens": 512,
    },
    "mage": {
        "description": "Longformer trained on MAGE corpus, 27 LLMs across 7 tasks (ACL 2024).",
        "window_tokens": 4096,
    },
    "radar": {
        "description": "RoBERTa-large adversarially trained against Vicuna-7B paraphrases (TrustSafeAI); paraphrase-robust.",
        "window_tokens": 512,
    },
    "binoculars": {
        "description": "Zero-shot detector via ratio of Falcon-7B perplexities (Hans et al., 2024); bf16.",
        "window_tokens": 1024,
    },
}

# Routing: each name belongs to exactly one GPU service.
TRAINED_NAMES = [
    "fakespot", "szeged", "desklib",
    "superannotate_low_fpr", "superannotate",
    "mage", "radar",
]
BINOCULARS_NAMES = ["binoculars"]

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

detector_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch",
        "transformers",
        "accelerate",
        "huggingface_hub",
        "numpy",
        "pydantic",
        "markdown",
        "beautifulsoup4",
        "safetensors",
    )
    .add_local_dir(".", remote_path="/root")
)

# Binoculars runs the dual Falcon-7B pair in fp16 on a 40 GB GPU.
binoculars_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch",
        "transformers",
        "accelerate",
        "huggingface_hub",
        "numpy",
        "pydantic",
    )
    .add_local_dir(".", remote_path="/root")
)

# CPU-only dispatcher image: just FastAPI + pydantic.
web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("fastapi", "pydantic")
)

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class DetectRequest(BaseModel):
    text: str
    models: list[str] | None = None

    @field_validator("text")
    @classmethod
    def _check_length(cls, v: str) -> str:
        words = len(v.split())
        if words < MIN_WORDS or words > MAX_WORDS:
            raise ValueError(
                f"Text has {words} words. Need {MIN_WORDS}-{MAX_WORDS} words "
                f"(word count is whitespace-split, approximate for CJK)."
            )
        return v


class DetectResponse(BaseModel):
    results: dict


# ---------------------------------------------------------------------------
# Trained-detector service (GPU)
# ---------------------------------------------------------------------------

@app.cls(
    image=detector_image,
    gpu=["L4", "A10G"],
    volumes={"/hf-cache": hf_cache},
    timeout=600,
    scaledown_window=20 * 60,
    max_containers=1,
)
class DetectorService:
    @modal.enter()
    def load(self):
        import os
        os.environ["HF_HOME"] = "/hf-cache"

        import torch
        from detectors import (
            DesklibDetector,
            FakespotDetector,
            MageDetector,
            RadarDetector,
            SuperAnnotateDetector,
            SuperAnnotateLowFprDetector,
            SzegedDetector,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        self.detectors = {
            d.name: d
            for d in [
                FakespotDetector(device),
                SzegedDetector(device),
                DesklibDetector(device),
                SuperAnnotateLowFprDetector(device),
                SuperAnnotateDetector(device),
                MageDetector(device),
                RadarDetector(device),
            ]
        }
        for d in list(self.detectors.values()):
            print(f"Loading {d.name}...")
            try:
                d.load()
                print(f"{d.name} ready.")
            except Exception as e:
                print(f"FAILED to load {d.name}: {e}")
                del self.detectors[d.name]

    @modal.method()
    def detect(self, text: str, names: list[str]) -> dict:
        results = {}
        for name in names:
            if name in self.detectors:
                results[name] = self.detectors[name].predict(text).model_dump()
        return results


# ---------------------------------------------------------------------------
# Binoculars service (GPU, separate image with bitsandbytes)
# ---------------------------------------------------------------------------

@app.cls(
    image=binoculars_image,
    gpu=["A100", "L40S"],
    volumes={"/hf-cache": hf_cache},
    timeout=900,
    scaledown_window=10 * 60,
    max_containers=1,
)
class BinocularsService:
    @modal.enter()
    def load(self):
        import os
        os.environ["HF_HOME"] = "/hf-cache"

        import torch
        from binoculars import BinocularsDetector

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Binoculars on {device}...")
        self.detector = BinocularsDetector(device)
        self.detector.load()
        print("Binoculars ready.")

    @modal.method()
    def detect(self, text: str) -> dict:
        return {"binoculars": self.detector.predict(text).model_dump()}


# ---------------------------------------------------------------------------
# Web dispatcher (no GPU)
# ---------------------------------------------------------------------------

@app.function(image=web_image, scaledown_window=20 * 60)
@modal.asgi_app(requires_proxy_auth=True)
def web():
    from fastapi import FastAPI, HTTPException

    api = FastAPI(title="Aletheia Detector")

    @api.get("/health")
    async def health():
        return {"status": "ok", "models": list(MODELS_META.keys())}

    @api.get("/models")
    async def list_models():
        return {
            "score_format": "P(AI) in [0,1]; >= 0.5 means AI Generated",
            "models": [
                {"name": name, **meta} for name, meta in MODELS_META.items()
            ],
        }

    @api.post("/detect", response_model=DetectResponse)
    async def detect(req: DetectRequest):
        import asyncio

        names = req.models or list(MODELS_META.keys())
        unknown = [n for n in names if n not in MODELS_META]
        if unknown:
            raise HTTPException(400, detail=f"Unknown models: {unknown}")

        trained = [n for n in names if n in TRAINED_NAMES]
        bino = [n for n in names if n in BINOCULARS_NAMES]

        # Run the two services concurrently via Modal's async interface.
        # A request that names only one kind never wakes the other.
        tasks = []
        if trained:
            tasks.append(DetectorService().detect.remote.aio(req.text, trained))
        if bino:
            tasks.append(BinocularsService().detect.remote.aio(req.text))

        results = {}
        for r in await asyncio.gather(*tasks):
            results.update(r)
        return DetectResponse(results=results)

    return api


# ---------------------------------------------------------------------------
# Local CLI (one-model smoke test)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    text: str = "This is a test sentence generated by an AI model to demonstrate capabilities.",
    model: str = "fakespot",
):
    print(f"Testing model: {model}")
    result = DetectorService().detect.remote(text, [model])
    print(f"Result: {result}")
