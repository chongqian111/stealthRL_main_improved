"""
Local reward utilities for the server-side 0.5B StealthRL workflow.

This module is shared by:
- train_local.py: GRPO reward function
- train_dpo_warmup.py: candidate ranking for preference pairs

The default reward is a threshold/margin objective aligned with ASR@1%FPR:
each detector is compared against a human-calibrated threshold, then the
reward combines the weighted mean margin and the worst detector margin.
"""

from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def make_prompt(text: str) -> str:
    """Build the paraphrase prompt used by local training/inference."""
    word_count = len(str(text).split())
    return (
        f"Paraphrase the following text while preserving its meaning. "
        f"Write approximately {word_count} words.\n\n"
        f"Original: {text}\n\n"
        f"Paraphrase:"
    )


def completion_to_text(completion: Any) -> str:
    """Normalize TRL string/chat completions to plain text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        if completion and isinstance(completion[-1], dict):
            return str(completion[-1].get("content", ""))
        return " ".join(str(x) for x in completion)
    return str(completion)


class LocalRewardScorer:
    """Detector + semantic scorer for local StealthRL training."""

    def __init__(self, cfg: dict[str, Any], tokenizer: Any | None = None):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.rob_model = None
        self.rob_tok = None
        self.fdgpt_tok = None
        self.fdgpt_model = None
        self.sem_model = None
        self.embed_cache: dict[str, torch.Tensor] = {}
        self.thresholds: dict[str, float] | None = None

    @property
    def detector_names(self) -> tuple[str, str]:
        return ("roberta", "fdgpt")

    def load_detectors(self) -> None:
        print("[detectors] Loading RoBERTa + Fast-DetectGPT scorers...")
        self.rob_tok = AutoTokenizer.from_pretrained(self.cfg["roberta_model"])
        self.rob_model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg["roberta_model"],
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self.rob_model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.rob_model.eval()
        print(f"  RoBERTa ready: {self.cfg['roberta_model']}")

        self.fdgpt_tok = AutoTokenizer.from_pretrained(self.cfg["fdgpt_model"])
        if self.fdgpt_tok.pad_token is None:
            self.fdgpt_tok.pad_token = self.fdgpt_tok.eos_token
        self.fdgpt_model = AutoModelForCausalLM.from_pretrained(
            self.cfg["fdgpt_model"],
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.fdgpt_model.eval()
        print(f"  Fast-DetectGPT scoring model ready: {self.cfg['fdgpt_model']}")

    def load_semantic_model(self) -> None:
        print("[semantic] Loading semantic similarity model...")
        try:
            self.sem_model = SentenceTransformer(
                self.cfg["semantic_model"],
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            print(f"  Semantic model ready: {self.cfg['semantic_model']}")
        except Exception as exc:
            print(f"  Failed to load {self.cfg['semantic_model']}: {exc}")
            self.sem_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("  Fallback semantic model ready: all-MiniLM-L6-v2")

    def close(self) -> None:
        """Best-effort cleanup before loading another training model."""
        self.rob_model = None
        self.rob_tok = None
        self.fdgpt_tok = None
        self.fdgpt_model = None
        self.sem_model = None
        self.embed_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def precompute_original_embeddings(self, train_data: list[dict[str, Any]]) -> None:
        if self.sem_model is None:
            self.load_semantic_model()

        originals = sorted({str(row.get("original_text", "")) for row in train_data})
        originals = [text for text in originals if text]
        print(f"[semantic] Precomputing E5 embeddings: {len(originals)} originals")
        for i in range(0, len(originals), self.cfg["semantic_batch_size"]):
            batch = originals[i : i + self.cfg["semantic_batch_size"]]
            embs = self.sem_model.encode(
                batch,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for text, emb in zip(batch, embs, strict=True):
                self.embed_cache[text] = emb.cpu()

    def score_roberta(self, texts: list[str]) -> list[float]:
        if self.rob_model is None:
            self.load_detectors()
        device = self.rob_model.device
        scores: list[float] = []
        for i in range(0, len(texts), self.cfg["roberta_batch_size"]):
            batch = texts[i : i + self.cfg["roberta_batch_size"]]
            inputs = self.rob_tok(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(device)
            with torch.no_grad():
                probs = torch.softmax(self.rob_model(**inputs).logits, dim=-1)
            # Project convention: class 0 = Fake/AI for OpenAI detector.
            scores.extend(probs[:, 0].detach().float().cpu().tolist())
        return scores

    def score_fast_detectgpt(self, texts: list[str]) -> list[float]:
        if self.fdgpt_model is None:
            self.load_detectors()

        scores: list[float] = []
        for i in range(0, len(texts), self.cfg["fdgpt_batch_size"]):
            batch = texts[i : i + self.cfg["fdgpt_batch_size"]]
            for text in batch:
                try:
                    inputs = self.fdgpt_tok(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    )
                    device = self.fdgpt_model.device
                    inputs = {key: value.to(device) for key, value in inputs.items()}
                    with torch.no_grad():
                        loss = self.fdgpt_model(
                            **inputs,
                            labels=inputs["input_ids"],
                        ).loss
                    scores.append(-float(loss.item()))
                except Exception:
                    scores.append(0.0)

        arr = torch.sigmoid(torch.tensor(scores, dtype=torch.float32) * 0.5)
        return arr.tolist()

    def detector_scores(self, texts: list[str]) -> dict[str, list[float]]:
        return {
            "roberta": self.score_roberta(texts),
            "fdgpt": self.score_fast_detectgpt(texts),
        }

    def _calibration_human_texts(self) -> list[str]:
        print("[thresholds] Loading MAGE human texts for detector calibration...")
        ds = load_dataset("yaful/MAGE", split=self.cfg["threshold_dataset_split"])
        texts: list[str] = []
        for row in ds:
            if row.get("label") != 1:
                continue
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            if self.tokenizer is not None:
                n_tok = len(self.tokenizer.encode(text))
                if not (self.cfg["token_min"] <= n_tok <= self.cfg["token_max"]):
                    continue
            texts.append(text)
            if len(texts) >= self.cfg["threshold_num_human"]:
                break
        if not texts:
            raise RuntimeError("No human calibration samples found.")
        print(f"  Calibration samples: {len(texts)}")
        return texts

    def load_or_calibrate_thresholds(self) -> dict[str, float]:
        path = Path(self.cfg["threshold_cache"])
        expected_meta = {
            "roberta_model": self.cfg["roberta_model"],
            "fdgpt_model": self.cfg["fdgpt_model"],
            "target_fpr": self.cfg["target_fpr"],
        }
        if path.exists():
            with path.open(encoding="utf-8") as f:
                payload = json.load(f)
            if payload.get("meta") == expected_meta:
                self.thresholds = {
                    key: float(value)
                    for key, value in payload["thresholds"].items()
                }
                print(f"[thresholds] Loaded cached thresholds: {path}")
                return self.thresholds
            print("[thresholds] Cache metadata changed; recalibrating.")

        texts = self._calibration_human_texts()
        scores = self.detector_scores(texts)
        percentile = 100.0 * (1.0 - float(self.cfg["target_fpr"]))
        thresholds = {
            name: float(torch.tensor(values).quantile(percentile / 100.0).item())
            for name, values in scores.items()
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": expected_meta,
            "thresholds": thresholds,
            "num_human": len(texts),
            "percentile": percentile,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self.thresholds = thresholds
        print(f"[thresholds] Saved calibrated thresholds: {path}")
        print(f"  thresholds={thresholds}")
        return thresholds

    def similarity(self, originals: list[str], completions: list[str]) -> list[float]:
        if self.sem_model is None:
            self.load_semantic_model()

        emb_a = torch.stack(
            [
                self.embed_cache.get(
                    text,
                    self.sem_model.encode(
                        text,
                        convert_to_tensor=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    ).cpu(),
                )
                for text in originals
            ]
        )
        emb_b = self.sem_model.encode(
            completions,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=self.cfg["semantic_batch_size"],
            show_progress_bar=False,
        )
        if torch.cuda.is_available():
            emb_a = emb_a.cuda()
            if emb_b.device.type != "cuda":
                emb_b = emb_b.cuda()
        return F.cosine_similarity(emb_a, emb_b).cpu().tolist()

    def _detector_reward(self, scores: dict[str, list[float]], idx: int) -> float:
        if self.cfg["reward_mode"] == "linear":
            ai_prob = (
                self.cfg["roberta_weight"] * scores["roberta"][idx]
                + self.cfg["fdgpt_weight"] * scores["fdgpt"][idx]
            )
            return 1.0 - ai_prob

        thresholds = self.thresholds or self.load_or_calibrate_thresholds()
        tau = max(float(self.cfg["margin_tau"]), 1e-6)
        margins = {}
        for name in self.detector_names:
            score = float(scores[name][idx])
            threshold = float(thresholds[name])
            margins[name] = 1.0 / (1.0 + math.exp(-(threshold - score) / tau))

        weighted = (
            self.cfg["roberta_weight"] * margins["roberta"]
            + self.cfg["fdgpt_weight"] * margins["fdgpt"]
        )
        worst = min(margins.values())
        return (
            (1.0 - self.cfg["worst_detector_weight"]) * weighted
            + self.cfg["worst_detector_weight"] * worst
        )

    def _length_penalty(self, original: str, completion: str) -> float:
        orig_words = max(len(original.split()), 1)
        comp_words = max(len(completion.split()), 1)
        ratio = comp_words / orig_words
        low = float(self.cfg["length_ratio_min"])
        high = float(self.cfg["length_ratio_max"])
        if low <= ratio <= high:
            return 0.0
        if ratio < low:
            distance = low - ratio
        else:
            distance = ratio - high
        return float(self.cfg["length_penalty_weight"]) * distance

    def compute_rewards(
        self,
        originals: list[str],
        completions: list[Any],
        return_details: bool = False,
    ) -> list[float] | tuple[list[float], list[dict[str, float]]]:
        clean = [completion_to_text(text).strip() for text in completions]
        rewards = [float(self.cfg["invalid_reward"])] * len(clean)
        details: list[dict[str, float]] = [{} for _ in clean]

        valid_idx: list[int] = []
        valid_originals: list[str] = []
        valid_completions: list[str] = []
        for idx, (original, completion) in enumerate(zip(originals, clean, strict=True)):
            if len(completion.split()) < int(self.cfg["min_completion_words"]):
                details[idx] = {"valid": 0.0}
                continue
            valid_idx.append(idx)
            valid_originals.append(str(original))
            valid_completions.append(completion)

        if not valid_idx:
            return (rewards, details) if return_details else rewards

        try:
            scores = self.detector_scores(valid_completions)
        except Exception as exc:
            print(f"  Detector scoring failed: {exc}")
            scores = {
                "roberta": [0.5] * len(valid_completions),
                "fdgpt": [0.5] * len(valid_completions),
            }

        try:
            sim_scores = self.similarity(valid_originals, valid_completions)
        except Exception as exc:
            print(f"  Semantic scoring failed: {exc}")
            sim_scores = [float(self.cfg["semantic_fallback"])] * len(valid_completions)

        for local_idx, global_idx in enumerate(valid_idx):
            original = valid_originals[local_idx]
            completion = valid_completions[local_idx]
            sim = float(sim_scores[local_idx])
            det_reward = self._detector_reward(scores, local_idx)
            length_penalty = self._length_penalty(original, completion)

            if sim < float(self.cfg["semantic_floor"]):
                semantic_term = -float(self.cfg["semantic_fail_penalty"])
            else:
                semantic_term = self.cfg["beta"] * (
                    (sim - self.cfg["semantic_floor"])
                    / max(1.0 - self.cfg["semantic_floor"], 1e-6)
                )

            reward = self.cfg["alpha"] * det_reward + semantic_term - length_penalty
            rewards[global_idx] = float(reward)
            details[global_idx] = {
                "valid": 1.0,
                "reward": float(reward),
                "detector_reward": float(det_reward),
                "semantic": sim,
                "length_penalty": float(length_penalty),
                "roberta": float(scores["roberta"][local_idx]),
                "fdgpt": float(scores["fdgpt"][local_idx]),
            }

        return (rewards, details) if return_details else rewards
