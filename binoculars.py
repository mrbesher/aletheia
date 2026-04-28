"""
Zero-shot AI text detection via Binoculars (Hans et al., NeurIPS 2024).

Score = perplexity_performer(text) / cross_entropy(observer, performer | text).
Lower scores indicate AI-generated text. We map the score to P(AI) using an
affine transform centred on the paper's accuracy-optimal threshold.

The cross-entropy term follows the reference implementation's `entropy()`:
no time-shift, averaged over all unpadded positions. Perplexity is the
standard shifted next-token NLL. The asymmetry is intentional and the
threshold is calibrated to this exact formulation.
"""

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Accuracy-optimal threshold for falcon-7b / falcon-7b-instruct from the paper.
BINOCULARS_THRESHOLD = 0.9015310749276843


class BinocularsResult(BaseModel):
    label: str
    score: float
    windows: int


def _perplexity(input_ids, attention_mask, logits):
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = input_ids[..., 1:].contiguous()
    shifted_mask = attention_mask[..., 1:].contiguous().float()

    losses = F.cross_entropy(
        shifted_logits.transpose(1, 2),
        shifted_labels,
        reduction="none",
    )
    masked = losses * shifted_mask
    return masked.sum(dim=-1) / shifted_mask.sum(dim=-1).clamp(min=1)


def _cross_entropy(observer_logits, performer_logits, attention_mask):
    mask = attention_mask.float()
    p_obs = F.softmax(observer_logits, dim=-1)
    log_p_perf = F.log_softmax(performer_logits, dim=-1)
    ce = -(p_obs * log_p_perf).sum(dim=-1)
    masked = ce * mask
    return masked.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)


class BinocularsDetector:
    name = "binoculars"
    description = (
        "Zero-shot detector via ratio of Falcon-7B perplexities "
        "(Hans et al., 2024); bf16."
    )
    max_tokens = 1024
    stride = 256
    threshold = BINOCULARS_THRESHOLD

    def __init__(self, device: str):
        self.device = device

    def load(self) -> None:
        observer_id = "tiiuae/falcon-7b"
        performer_id = "tiiuae/falcon-7b-instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(observer_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.observer = AutoModelForCausalLM.from_pretrained(
            observer_id,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
        )
        self.performer = AutoModelForCausalLM.from_pretrained(
            performer_id,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
        )
        self.observer.eval()
        self.performer.eval()

    def _chunks(self, text: str) -> list[str]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= self.max_tokens:
            return [text]
        step = self.max_tokens - self.stride
        out = []
        for start in range(0, len(ids), step):
            piece = ids[start : start + self.max_tokens]
            if not piece:
                break
            out.append(self.tokenizer.decode(piece, skip_special_tokens=True))
            if start + self.max_tokens >= len(ids):
                break
        return out

    @torch.inference_mode()
    def _score_window(self, text: str) -> float:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokens,
            padding=True,
        ).to(self.device)

        obs_logits = self.observer(**enc).logits
        perf_logits = self.performer(**enc).logits

        ppl = _perplexity(enc["input_ids"], enc["attention_mask"], perf_logits)
        x_ppl = _cross_entropy(obs_logits, perf_logits, enc["attention_mask"])
        score = (ppl / x_ppl).item()

        p_ai = (self.threshold - score) / self.threshold + 0.5
        return max(0.0, min(1.0, p_ai))

    def predict(self, text: str) -> BinocularsResult:
        windows = self._chunks(text)
        scores = [self._score_window(w) for w in windows]
        if not scores:
            return BinocularsResult(label="unknown", score=float("nan"), windows=0)
        avg = sum(scores) / len(scores)
        label = "AI Generated" if avg >= 0.5 else "Human"
        return BinocularsResult(label=label, score=avg, windows=len(windows))
