# train.py
# -*- coding: utf-8 -*-

"""
Training script for LMOL on SCUT-FBP5500 with QLoRA.
Key changes to fully match the paper:
- Dual learning rates: projection matrix W (2e-5) vs. LoRA (2e-4).
- Cosine annealing schedulers to eta_min=0 over total steps (step-wise).
- Safe collate_fn to pad input_ids/labels and build attention_mask.
- Freeze CLIP; train full projection; train only LoRA params in the LLM.
All comments are in English as requested.
"""

import os
from typing import Iterable, Dict, Any

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms

from transformers import AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

from models.lmol_model import LMOLModel
from datasets.scut_fbp_dataset import SCUTFBPDataset
from utils.collate import lmol_collate_fn


def count_steps(dataloader: DataLoader, num_epochs: int) -> int:
    """Compute total optimization steps."""
    return len(dataloader) * num_epochs


def separate_params_for_optim(model: LMOLModel) -> Dict[str, Iterable[torch.nn.Parameter]]:
    """
    Split trainable parameters into two groups:
      - 'proj': projection matrix W (full finetuning)
      - 'lora': LoRA-injected params in the LLM
    """
    proj_params = []
    lora_params = []

    # Projection matrix W
    for n, p in model.projection.named_parameters():
        if p.requires_grad:
            proj_params.append(p)

    # LoRA params in the language model
    for n, p in model.language_model.named_parameters():
        # PEFT names contain "lora"
        if p.requires_grad and "lora" in n:
            lora_params.append(p)

    return {"proj": proj_params, "lora": lora_params}


def main() -> None:
    # -----------------------------
    # Basic configuration
    # -----------------------------
    vision_model_name = "openai/clip-vit-large-patch14-336"  # CLIP ViT-L/14 336px
    llm_model_name = "lmsys/vicuna-7b-v1.5"                 # Vicuna-1.5 7B
    num_epochs = 1
    train_batch_size = 4
    # Dual-LR per paper
    lr_proj = 2e-5   # projection matrix W
    lr_lora = 2e-4   # LoRA params in LLM

    # -----------------------------
    # Tokenizer
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # -----------------------------
    # Data
    # -----------------------------
    transform = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])
    train_dataset = SCUTFBPDataset(
        split='train', download=True, transform=transform, tokenizer=tokenizer
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=lmol_collate_fn,
    )

    # -----------------------------
    # Model (QLoRA + frozen CLIP)
    # -----------------------------
    # 4-bit quantization config for Vicuna
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = LMOLModel(vision_model_name=vision_model_name, llm_model_name=llm_model_name)
    model = model.to(device)

    # Apply LoRA to language model (attention projections etc.)
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=32,          # corresponds to scaling λ=4 in the paper (alpha / r)
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model.language_model = get_peft_model(model.language_model, lora_cfg)
    model.train()

    # Ensure trainable flags are correct
    for p in model.visual_encoder.parameters():
        p.requires_grad = False
    for n, p in model.language_model.named_parameters():
        # Only LoRA params should be trainable inside the LLM
        p.requires_grad = ("lora" in n)
    for p in model.projection.parameters():
        p.requires_grad = True

    # -----------------------------
    # Optimizer with param groups
    # -----------------------------
    param_groups = separate_params_for_optim(model)
    optimizer = AdamW(
        [
            {"params": param_groups["proj"], "lr": lr_proj},
            {"params": param_groups["lora"], "lr": lr_lora},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )

    # -----------------------------
    # Cosine annealing schedulers (step-wise) to eta_min=0
    # -----------------------------
    total_steps = count_steps(train_loader, num_epochs)
    # One scheduler object can handle multiple param groups with different base LRs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0.0)

    # -----------------------------
    # Training loop
    # -----------------------------
    global_step = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in train_loader:
            img1 = batch["img1"].to(device, non_blocking=True)
            img2 = batch["img2"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                image1_tensor=img1,
                image2_tensor=img2,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()  # step-wise cosine annealing

            running_loss += loss.item()
            global_step += 1

            if global_step % 50 == 0:
                # Print current LRs for both param groups
                curr_lrs = [g["lr"] for g in optimizer.param_groups]
                print(f"[Step {global_step}/{total_steps}] loss={loss.item():.4f} | lr_proj={curr_lrs[0]:.6e} lr_lora={curr_lrs[1]:.6e}")

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. avg_loss={epoch_loss:.4f}")

    # Save (you may want to save LoRA adapters separately)
    torch.save(model.state_dict(), "lmol_model.pt")
    print("Training completed and model saved to lmol_model.pt")


if __name__ == "__main__":
    main()
