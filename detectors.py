import abc
import re
from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    pipeline,
)


class DetectorResult(BaseModel):
    label: str
    score: float
    windows: int


def _chunk_text(text: str, tokenizer, max_tokens: int, stride: int) -> List[str]:
    if not text or not text.strip():
        return []
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return [text]
    step = max_tokens - stride
    windows = []
    for start in range(0, len(token_ids), step):
        end = start + max_tokens
        piece = token_ids[start:end]
        if not piece:
            break
        windows.append(tokenizer.decode(piece, skip_special_tokens=True))
        if end >= len(token_ids):
            break
    return windows


def _aggregate(scores: List[float], threshold: float) -> tuple[float, str]:
    if not scores:
        return float("nan"), "unknown"
    doc_score = float(np.mean(scores))
    label = "AI Generated" if doc_score >= threshold else "Human"
    return doc_score, label


class BaseDetector(abc.ABC):
    name: str
    threshold: float = 0.5

    def __init__(self, device: str):
        self.device = device

    @abc.abstractmethod
    def load(self) -> None:
        ...

    @abc.abstractmethod
    def predict(self, text: str) -> DetectorResult:
        ...


class _ChunkingDetector(BaseDetector):
    max_tokens: int
    stride: int
    tokenizer: Any

    def predict(self, text: str) -> DetectorResult:
        windows = _chunk_text(text, self.tokenizer, self.max_tokens, self.stride)
        scores = [self._predict_window(w) for w in windows]
        score, label = _aggregate(scores, self.threshold)
        return DetectorResult(label=label, score=score, windows=len(windows))

    def _predict_window(self, text: str) -> float:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# 1) Fakespot — standard HF pipeline
# --------------------------------------------------------------------------- #

class FakespotDetector(_ChunkingDetector):
    name = "fakespot"
    max_tokens = 512
    stride = 128

    def load(self) -> None:
        model_id = "fakespot-ai/roberta-base-ai-text-detection-v1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.pipe = pipeline(
            "text-classification",
            model=model_id,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            truncation=True,
            max_length=self.max_tokens,
        )

    @staticmethod
    def _clean(text: str) -> str:
        text = text.replace("\u200b", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _predict_window(self, text: str) -> float:
        cleaned = self._clean(text)
        out = self.pipe(cleaned)[0]
        label = str(out["label"]).lower()
        score = float(out["score"])
        if "ai" in label or label in {"label_1", "generated", "fake"}:
            return score
        return 1.0 - score if score <= 1.0 else score


# --------------------------------------------------------------------------- #
# 2) SzegedAI — ModernBERT ensemble (3 checkpoints)
# --------------------------------------------------------------------------- #

class SzegedDetector(_ChunkingDetector):
    name = "szeged"
    max_tokens = 2048
    stride = 256

    def load(self) -> None:
        from huggingface_hub import hf_hub_download

        base_id = "answerdotai/ModernBERT-base"
        self.tokenizer = AutoTokenizer.from_pretrained(base_id)

        ckpt1 = hf_hub_download(
            repo_id="SzegedAI/AI_Detector",
            filename="modernbert.bin",
            repo_type="space",
        )
        ckpt2 = hf_hub_download(
            repo_id="mihalykiss/modernbert_2",
            filename="Model_groups_3class_seed12",
        )
        ckpt3 = hf_hub_download(
            repo_id="mihalykiss/modernbert_2",
            filename="Model_groups_3class_seed22",
        )

        num_labels = 41
        self.models = []
        for ckpt in [ckpt1, ckpt2, ckpt3]:
            model = AutoModelForSequenceClassification.from_pretrained(
                base_id, num_labels=num_labels
            )
            try:
                state = torch.load(ckpt, map_location=self.device, weights_only=True)
            except Exception:
                state = torch.load(ckpt, map_location=self.device, weights_only=False)
            model.load_state_dict(state, strict=True)
            model.to(self.device).eval()
            self.models.append(model)

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"\s+([,.;:?!])", r"\1", text)
        return text.strip()

    @torch.inference_mode()
    def _predict_window(self, text: str) -> float:
        text = self._clean(text)
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_tokens,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        probs = None
        for model in self.models:
            logits = model(**enc).logits
            if probs is None:
                probs = torch.softmax(logits, dim=-1)
            else:
                probs += torch.softmax(logits, dim=-1)
        probs = probs / len(self.models)

        human_prob = float(probs.squeeze(0)[24].item())
        return 1.0 - human_prob


# --------------------------------------------------------------------------- #
# 3) Desklib — custom PreTrainedModel subclass
# --------------------------------------------------------------------------- #

class DesklibDetector(_ChunkingDetector):
    name = "desklib"
    max_tokens = 768
    stride = 192

    class _Model(PreTrainedModel):
        config_class = AutoConfig

        def __init__(self, config):
            super().__init__(config)
            self.model = AutoModel.from_config(config)
            self.classifier = nn.Linear(config.hidden_size, 1)
            self.init_weights()

        @property
        def all_tied_weights_keys(self):
            return {}

        def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
            outputs = self.model(input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs[0]
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            )
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
            logits = self.classifier(pooled_output)
            loss = None
            if labels is not None:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1), labels.float())
            out = {"logits": logits}
            if loss is not None:
                out["loss"] = loss
            return out

    def load(self) -> None:
        model_id = "desklib/ai-text-detector-v1.01"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = self._Model.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def _predict_window(self, text: str) -> float:
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        outputs = self.model(**enc)
        prob_ai = torch.sigmoid(outputs["logits"]).item()
        return float(prob_ai)


# --------------------------------------------------------------------------- #
# 4) SuperAnnotate — RoBERTa-large with custom single-logit head (RAID #1 OSS)
# --------------------------------------------------------------------------- #

# Cyrillic/Greek -> ASCII lookalikes, copied from SuperAnnotate's preprocessing
# https://github.com/superannotateai/generated_text_detector
_HOMOGLYPH_MAP = {
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "Х": "X", "а": "a", "е": "e", "о": "o",
    "р": "p", "с": "c", "у": "y", "х": "x", "І": "I", "і": "i",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    "Ϲ": "C", "ο": "o",
}
_URL_PATTERN = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")


def _superannotate_preprocess(text: str) -> str:
    import markdown
    from bs4 import BeautifulSoup

    text = text.replace("​", "")
    text = "".join(_HOMOGLYPH_MAP.get(c, c) for c in text)
    html = markdown.markdown(text)
    text = BeautifulSoup(html, "html.parser").get_text()
    text = _URL_PATTERN.sub("", text)
    text = _EMAIL_PATTERN.sub("", text)
    return " ".join(text.split()).strip()


class _SuperAnnotateRoberta(nn.Module):
    """Mirrors SuperAnnotate's RobertaClassifier: RoBERTa (no pooler) + dropout + Linear(H, 1)."""

    def __init__(self, base_id: str, classifier_dropout: float):
        super().__init__()
        from transformers import RobertaConfig, RobertaModel

        cfg = RobertaConfig.from_pretrained(base_id)
        self.roberta = RobertaModel(cfg, add_pooling_layer=False)
        self.dropout = nn.Dropout(classifier_dropout)
        self.dense = nn.Linear(cfg.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, **_):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        x = out[0][:, 0, :]  # <s> / CLS
        x = self.dropout(x)
        return self.dense(x)


class _SuperAnnotateBase(_ChunkingDetector):
    max_tokens = 512
    stride = 128
    model_id: str = ""

    def predict(self, text: str) -> DetectorResult:
        return super().predict(_superannotate_preprocess(text))

    def load(self) -> None:
        import json

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        config_path = hf_hub_download(repo_id=self.model_id, filename="config.json")
        with open(config_path) as f:
            cfg = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = _SuperAnnotateRoberta(
            base_id=cfg["pretrain_checkpoint"],
            classifier_dropout=cfg.get("classifier_dropout", 0.1),
        )
        weights_path = hf_hub_download(repo_id=self.model_id, filename="model.safetensors")
        state = load_file(weights_path)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def _predict_window(self, text: str) -> float:
        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_tokens,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items() if k != "token_type_ids"}
        logits = self.model(**enc)
        return float(torch.sigmoid(logits).squeeze().item())


class SuperAnnotateLowFprDetector(_SuperAnnotateBase):
    name = "superannotate_low_fpr"
    model_id = "SuperAnnotate/ai-detector-low-fpr"


class SuperAnnotateDetector(_SuperAnnotateBase):
    name = "superannotate"
    model_id = "SuperAnnotate/ai-detector"


# --------------------------------------------------------------------------- #
# 5) MAGE — Longformer for long-context machine-text detection (ACL 2024)
#    Per training script: label 0 = AI, label 1 = human  (inverted from convention)
# --------------------------------------------------------------------------- #

class MageDetector(_ChunkingDetector):
    name = "mage"
    max_tokens = 4096
    stride = 512

    def load(self) -> None:
        import json

        from huggingface_hub import hf_hub_download

        model_id = "yaful/MAGE"
        # MAGE's published config has int-valued id2label which fails strict
        # validation in recent transformers. Sanitize the cached config.json
        # in place before letting transformers parse it.
        config_path = hf_hub_download(model_id, "config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        i2l = cfg.get("id2label") or {}
        if any(not isinstance(v, str) for v in i2l.values()):
            cfg["id2label"] = {str(k): "AI" if int(k) == 0 else "human" for k in i2l}
            cfg["label2id"] = {v: int(k) for k, v in cfg["id2label"].items()}
            with open(config_path, "w") as f:
                json.dump(cfg, f)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(
            self.device
        )
        self.model.eval()

    @torch.inference_mode()
    def _predict_window(self, text: str) -> float:
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_tokens,
            padding=False,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits
        # MAGE: index 0 is AI, index 1 is human
        return float(torch.softmax(logits, dim=-1).squeeze(0)[0].item())


# --------------------------------------------------------------------------- #
# 6) RADAR — RoBERTa-large adversarially trained against a Vicuna-7B paraphraser
#    Despite the "Vicuna-7B" suffix, the detector itself is RoBERTa-large (355M).
#    Per their HF Space inference: index 0 = AI, index 1 = human
#    License: non-commercial (inherited from Vicuna)
# --------------------------------------------------------------------------- #

class RadarDetector(_ChunkingDetector):
    name = "radar"
    max_tokens = 512
    stride = 128

    def load(self) -> None:
        model_id = "TrustSafeAI/RADAR-Vicuna-7B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(
            self.device
        )
        self.model.eval()

    @torch.inference_mode()
    def _predict_window(self, text: str) -> float:
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_tokens,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits
        # RADAR: index 0 is AI
        return float(torch.softmax(logits, dim=-1).squeeze(0)[0].item())
