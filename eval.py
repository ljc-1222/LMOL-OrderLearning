#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation script for LMOL using the full Bradley–Terry ordinal logistic model.

This script estimates a continuous facial beauty score for each test image
based on pairwise comparisons with a set of reference images.  The
reference set is constructed dynamically from the training split by
discretising ground truth scores into bins of width ``score_interval`` and
sampling up to ``max_reference_per_bin`` images per bin.  For each test
image, the trained LMOL comparator predicts the ordinal relation (greater,
similar, or less attractive) between the test image and every reference
image.  These outcomes are combined via a Bradley–Terry ordinal logistic
model to estimate the test image’s score.

All hyperparameters controlling the reference set construction and BT
estimation (such as ``score_interval``, ``max_reference_per_bin``, ``bt_delta``
and ``bt_k``) are loaded from ``configs/lmol_config.json``.  The final
evaluation prints Pearson correlation (PC), mean absolute error (MAE) and
root mean squared error (RMSE) between predicted scores and ground truth
scores on the test split.
"""

import json
import math
import os
from typing import List, Tuple, Dict

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer
from torchvision import transforms

from datasets.scut_fbp_dataset import SCUTFBPDataset
from models.lmol_model import LMOLModel


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def estimate_score_bt(
    ref_scores: List[float], outcomes: List[str],
    *, delta: float, k: float,
    score_min: float, score_max: float,
    max_iters: int = 50, tol: float = 1e-5
) -> float:
    """Estimate the target score using the full Bradley–Terry ordinal logistic model.

    Given a set of reference scores and corresponding ordinal outcomes,
    this function finds the score ``s_t`` that maximizes the joint
    likelihood under the Bradley–Terry model described in the UOL paper.
    The ordinal outcomes are encoded as strings: "First" indicates
    ``s_t > s_r`` (label ``2``), "Second" indicates ``s_t < s_r`` (label
    ``0``) and "Similar" indicates an approximate tie (label ``1``).  Two
    parameters control the shape of the logistic curves: ``delta`` (which
    tunes the width of the tie region) and ``k`` (which scales how
    quickly probabilities change with score differences).  A Newton–Raphson
    optimisation is performed on the log‑likelihood.  The search is
    constrained to the range ``[score_min, score_max]``.

    Parameters
    ----------
    ref_scores: List[float]
        Ground‑truth scores for each reference image.
    outcomes: List[str]
        Ordinal outcomes predicted by the comparator.  Each element must
        be one of ``"First"``, ``"Second"`` or ``"Similar"``.
    delta: float
        Threshold parameter δ controlling the tie region in the ordinal
        logistic model.  Larger values increase the probability mass of
        the "Similar" category for small score differences.
    k: float
        Scaling parameter controlling the slope of the logistic curves.
    score_min: float
        Lower bound of the allowable score range.
    score_max: float
        Upper bound of the allowable score range.
    max_iters: int, optional
        Maximum number of Newton–Raphson iterations.  Defaults to 50.
    tol: float, optional
        Convergence tolerance for successive updates.  Defaults to 1e‑5.

    Returns
    -------
    float
        The estimated continuous beauty score for the target image.
    """
    s_t = sum(ref_scores) / len(ref_scores)  # initial guess at mid‑point
    for _ in range(max_iters):
        grad = 0.0
        hess = 0.0
        for s_r, outcome in zip(ref_scores, outcomes):
            sdiff = k * (s_t - s_r)
            f_minus = 1.0 / (1.0 + math.exp(-(sdiff - delta)))
            f_plus = 1.0 / (1.0 + math.exp(-(sdiff + delta)))
            df_minus = f_minus * (1.0 - f_minus) * k
            df_plus = f_plus * (1.0 - f_plus) * k
            if outcome == "Second":
                grad += k * (1.0 - f_minus)
                hess += - (k * k) * f_minus * (1.0 - f_minus)
            elif outcome == "First":
                grad += -k * f_plus
                hess += - (k * k) * f_plus * (1.0 - f_plus)
            else:  # "Similar"
                A = f_plus * (1.0 - f_plus)
                B = f_minus * (1.0 - f_minus)
                N = A - B
                C = f_plus - f_minus
                grad += k * N / C
                dA = (1.0 - 2.0 * f_plus) * A * k
                dB = (1.0 - 2.0 * f_minus) * B * k
                dN = dA - dB
                hess += k * ((dN / C) - (k * N * N) / (C * C))
        if abs(hess) < 1e-9:
            break
        s_new = s_t - (grad / hess)
        # Clamp to allowable range
        if s_new < score_min:
            s_new = score_min
        elif s_new > score_max:
            s_new = score_max
        if abs(s_new - s_t) < tol:
            s_t = s_new
            break
        s_t = s_new
    return max(min(s_t, score_max), score_min)


def main() -> None:
    cfg = load_config(os.path.join("configs", "lmol_config.json"))
    vision_model_name = cfg["vision_model_name"]
    llm_model_name = cfg["llm_model_name"]
    root_dir = cfg.get("root_dir", "data/SCUT-FBP5500")

    # Parameters for reference set construction and BT estimation
    interval = cfg.get("score_interval", 0.1)
    max_per_bin = cfg.get("max_reference_per_bin", 10)
    delta = cfg.get("bt_delta", 0.1)
    k = cfg.get("bt_k", 1.0)
    score_min = cfg.get("bt_score_min", None)
    score_max = cfg.get("bt_score_max", None)

    # Load the trained LMOL model
    model = LMOLModel(vision_model_name, llm_model_name)
    model.load_state_dict(torch.load("lmol_model.pt", map_location="cpu"))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Tokenizer (no special tokens besides EOS)
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # Image transform (must match training)
    transform = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    ])

    # Load datasets just for accessing image paths and scores
    train_dataset = SCUTFBPDataset(
        root_dir=root_dir,
        split="train",
        download=False,
        transform=transform,
        tokenizer=tokenizer,
        theta=cfg["theta"],
        M=cfg["M"],
    )
    test_dataset = SCUTFBPDataset(
        root_dir=root_dir,
        split="test",
        download=False,
        transform=transform,
        tokenizer=tokenizer,
        theta=cfg["theta"],
        M=cfg["M"],
    )

    # Set score bounds if not provided
    if score_min is None:
        score_min = float(np.min(train_dataset.scores))
    if score_max is None:
        score_max = float(np.max(train_dataset.scores))

    # Build the reference set from training data
    bins: Dict[float, List[int]] = {}
    for idx, score in enumerate(train_dataset.scores):
        discrete = round(score / interval) * interval
        bins.setdefault(discrete, []).append(idx)
    reference_indices: List[int] = []
    reference_scores: List[float] = []
    for bin_val in sorted(bins.keys()):
        indices = bins[bin_val]
        sample = indices[:max_per_bin]
        for idx in sample:
            reference_indices.append(idx)
            reference_scores.append(float(train_dataset.scores[idx]))
    # Precompute reference image tensors
    reference_images: List[torch.Tensor] = []
    for idx in reference_indices:
        img_path = train_dataset.image_paths[idx]
        pil_img = Image.open(img_path).convert("RGB")
        ref_tensor = transform(pil_img).unsqueeze(0).to(device)
        reference_images.append(ref_tensor)

    # Candidate token IDs for "First", "Second", "Similar"
    candidate_token_ids = [
        tokenizer.encode(" First", add_special_tokens=False)[0],
        tokenizer.encode(" Second", add_special_tokens=False)[0],
        tokenizer.encode(" Similar", add_special_tokens=False)[0],
    ]

    true_scores: List[float] = []
    predicted_scores: List[float] = []

    # Loop over each test image, perform comparisons and estimate score
    for i in range(len(test_dataset.scores)):
        test_img_path = test_dataset.image_paths[i]
        pil_img = Image.open(test_img_path).convert("RGB")
        img_tensor = transform(pil_img).unsqueeze(0).to(device)

        outcomes_list: List[str] = []
        for ref_tensor in reference_images:
            question_ids = tokenizer.encode(
                ", which face looks more attractive?", return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                out = model(
                    image1_tensor=img_tensor,
                    image2_tensor=ref_tensor,
                    input_ids=question_ids,
                    labels=None,
                )
                logits = out.logits
            last_logits = logits[0, -1, :]
            cand_logits = last_logits[candidate_token_ids]
            choice = torch.argmax(cand_logits).item()
            answer_id = candidate_token_ids[choice]
            answer_text = tokenizer.decode([answer_id]).strip()
            outcomes_list.append(answer_text)

        pred_score = estimate_score_bt(
            reference_scores, outcomes_list,
            delta=delta, k=k,
            score_min=score_min, score_max=score_max
        )
        true_scores.append(float(test_dataset.scores[i]))
        predicted_scores.append(pred_score)

    # Compute metrics
    true_arr = np.array(true_scores)
    pred_arr = np.array(predicted_scores)
    pc = np.corrcoef(true_arr, pred_arr)[0, 1]
    mae = float(np.mean(np.abs(pred_arr - true_arr)))
    rmse = float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)))
    print(f"Evaluation results - PC: {pc:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()
