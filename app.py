import asyncio
import os
from dataclasses import dataclass
from typing import Protocol

import modal
from pydantic import BaseModel, field_validator

MIN_WORDS = 40
MAX_WORDS = 5000


@dataclass(frozen=True)
class Model:
    description: str
    window_tokens: int
    engine: str


MODELS: dict[str, Model] = {
    "fakespot": Model("RoBERTa-base binary AI detector (Fakespot).", 512, "trained"),
    "szeged": Model("Three-seed ModernBERT ensemble, 41-class LLM-family classifier (SzegedAI); reports 1 minus P(human).", 2048, "trained"),
    "desklib": Model("Mean-pooled transformer with single-logit AI head (Desklib).", 768, "trained"),
    "superannotate_low_fpr": Model("RoBERTa-large detector tuned for low false-positive rate (SuperAnnotate).", 512, "trained"),
    "superannotate": Model("RoBERTa-large balanced AI detector (SuperAnnotate).", 512, "trained"),
    "mage": Model("Longformer trained on MAGE corpus, 27 LLMs across 7 tasks (ACL 2024).", 4096, "trained"),
    "radar": Model("RoBERTa-large adversarially trained against Vicuna-7B paraphrases (TrustSafeAI); paraphrase-robust.", 512, "trained"),
    "binoculars": Model("Zero-shot detector via ratio of Falcon-7B perplexities (Hans et al., 2024); bf16.", 1024, "binoculars"),
}


class DetectRequest(BaseModel):
    text: str
    models: list[str] | None = None
    verbose: bool = False

    @field_validator("text")
    @classmethod
    def _word_count(cls, v: str) -> str:
        words = len(v.split())
        if not MIN_WORDS <= words <= MAX_WORDS:
            raise ValueError(
                f"Text has {words} words. Need {MIN_WORDS}-{MAX_WORDS} words "
                f"(word count is whitespace-split, approximate for CJK)."
            )
        return v


class DetectResponse(BaseModel):
    results: dict


def load_trained(device: str) -> dict:
    from detectors import (
        DesklibDetector,
        FakespotDetector,
        MageDetector,
        RadarDetector,
        SuperAnnotateDetector,
        SuperAnnotateLowFprDetector,
        SzegedDetector,
    )

    classes = [
        FakespotDetector,
        SzegedDetector,
        DesklibDetector,
        SuperAnnotateLowFprDetector,
        SuperAnnotateDetector,
        MageDetector,
        RadarDetector,
    ]
    loaded: dict = {}
    for cls in classes:
        d = cls(device)
        d.load()
        loaded[d.name] = d
    return loaded


def load_binoculars(device: str):
    from binoculars import BinocularsDetector

    detector = BinocularsDetector(device)
    detector.load()
    return detector


def split(names: list[str]) -> tuple[list[str], list[str]]:
    trained = [n for n in names if MODELS[n].engine == "trained"]
    binoculars = [n for n in names if MODELS[n].engine == "binoculars"]
    return trained, binoculars


def run_trained(detectors: dict, text: str, names: list[str], verbose: bool = False) -> dict:
    return {n: detectors[n].predict(text, verbose).model_dump() for n in names if n in detectors}


def run_binoculars(detector, text: str, verbose: bool = False) -> dict:
    return {"binoculars": detector.predict(text, verbose).model_dump()}


class Engine(Protocol):
    async def detect(self, text: str, names: list[str], verbose: bool = False) -> dict: ...


class LocalEngine:
    def __init__(self, device: str = "cuda") -> None:
        self.trained = load_trained(device)
        self.binoculars = load_binoculars(device)

    async def detect(self, text: str, names: list[str], verbose: bool = False) -> dict:
        trained, binoculars = split(names)
        return {
            **run_trained(self.trained, text, trained, verbose),
            **(run_binoculars(self.binoculars, text, verbose) if binoculars else {}),
        }


def create_api(engine: Engine):
    from fastapi import FastAPI, HTTPException

    api = FastAPI(title="Aletheia Detector")

    @api.get("/health")
    async def health():
        return {"status": "ok", "models": list(MODELS)}

    @api.get("/models")
    async def models():
        return {
            "score_format": "P(AI) in [0,1]; >= 0.5 means AI Generated",
            "models": [
                {"name": name, "description": m.description, "window_tokens": m.window_tokens}
                for name, m in MODELS.items()
            ],
        }

    @api.post("/detect", response_model=DetectResponse)
    async def detect(req: DetectRequest):
        names = req.models or list(MODELS)
        unknown = [n for n in names if n not in MODELS]
        if unknown:
            raise HTTPException(400, f"Unknown models: {unknown}")
        return DetectResponse(results=await engine.detect(req.text, names, req.verbose))

    return api


app = modal.App("aletheia-detector")
hf_cache = modal.Volume.from_name("aletheia-hf-cache", create_if_missing=True)

gpu_image = (
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
    .add_local_python_source("app", "detectors", "binoculars")
)

web_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("fastapi", "pydantic")


@app.cls(
    image=gpu_image,
    gpu=["L4", "A10G"],
    volumes={"/hf-cache": hf_cache},
    timeout=600,
    scaledown_window=20 * 60,
    max_containers=1,
)
class TrainedService:
    @modal.enter()
    def setup(self):
        os.environ["HF_HOME"] = "/hf-cache"
        self.detectors = load_trained("cuda")

    @modal.method()
    def detect(self, text: str, names: list[str], verbose: bool = False) -> dict:
        return run_trained(self.detectors, text, names, verbose)


@app.cls(
    image=gpu_image,
    gpu=["A100", "L40S"],
    volumes={"/hf-cache": hf_cache},
    timeout=900,
    scaledown_window=10 * 60,
    max_containers=1,
)
class BinocularsService:
    @modal.enter()
    def setup(self):
        os.environ["HF_HOME"] = "/hf-cache"
        self.detector = load_binoculars("cuda")

    @modal.method()
    def detect(self, text: str, verbose: bool = False) -> dict:
        return run_binoculars(self.detector, text, verbose)


class ModalEngine:
    async def detect(self, text: str, names: list[str], verbose: bool = False) -> dict:
        trained, binoculars = split(names)
        calls = []
        if trained:
            calls.append(TrainedService().detect.remote.aio(text, trained, verbose))
        if binoculars:
            calls.append(BinocularsService().detect.remote.aio(text, verbose))
        merged: dict = {}
        for partial in await asyncio.gather(*calls):
            merged.update(partial)
        return merged


@app.function(image=web_image, scaledown_window=20 * 60)
@modal.asgi_app(requires_proxy_auth=True)
def web():
    return create_api(ModalEngine())


@app.local_entrypoint()
def smoke(text: str = "The quick brown fox jumps over the lazy dog. " * 5, model: str = "fakespot"):
    service = TrainedService if MODELS[model].engine == "trained" else BinocularsService
    args = (text, [model]) if MODELS[model].engine == "trained" else (text,)
    print(service().detect.remote(*args))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_api(LocalEngine()), host="0.0.0.0", port=8000)
