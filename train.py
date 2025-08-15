#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from typing import Iterable, Dict

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms

from transformers import AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

from datasets.scut_fbp_dataset import SCUTFBPDataset
from models.lmol_model import LMOLModel
from utils.collate import lmol_collate_fn

def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return json.load(f)

def count_steps(dataloader: DataLoader, num_epochs: int) -> int:
    return len(dataloader) * num_epochs

def separate_params_for_optim(model: LMOLModel) -> Dict[str, Iterable[torch.nn.Parameter]]:
    proj_params = []
    lora_params = []
    for _, p in model.projection.named_parameters():
        if p.requires_grad:
            proj_params.append(p)
    for name, p in model.language_model.named_parameters():
        if p.requires_grad and "lora" in name:
            lora_params.append(p)
    return {"proj": proj_params, "lora": lora_params}

def main() -> None:
    cfg = load_config(os.path.join("configs", "lmol_config.json"))
    vision_model_name = cfg["vision_model_name"]
    llm_model_name = cfg["llm_model_name"]
    num_epochs = cfg["num_epochs"]
    train_batch_size = cfg["train_batch_size"]
    lr_proj = cfg["lr_proj"]
    lr_lora = cfg["lr_lora"]
    theta = cfg["theta"]
    M = cfg["M"]
    root_dir = cfg.get("root_dir", "data/SCUT-FBP5500")

    tokenizer = AutoTokenizer.from_pretrained(llm_model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    transform = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    train_dataset = SCUTFBPDataset(
        root_dir=root_dir,
        split="train",
        download=True,
        transform=transform,
        tokenizer=tokenizer,
        theta=theta,
        M=M,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=lmol_collate_fn,
    )

    model = LMOLModel(vision_model_name=vision_model_name, llm_model_name=llm_model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model.language_model = get_peft_model(model.language_model, lora_cfg)
    model.train()
    for p in model.visual_encoder.parameters():
        p.requires_grad = False
    for name, p in model.language_model.named_parameters():
        p.requires_grad = ("lora" in name)
    for p in model.projection.parameters():
        p.requires_grad = True

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

    total_steps = count_steps(train_loader, num_epochs)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0.0)

    global_step = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            img1 = batch["img1"].to(device, non_blocking=True)
            img2 = batch["img2"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
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
            scheduler.step()
            running_loss += loss.item()
            global_step += 1
            if global_step % 50 == 0:
                curr_lrs = [g["lr"] for g in optimizer.param_groups]
                print(f"[Step {global_step}/{total_steps}] loss={loss.item():.4f} | lr_proj={curr_lrs[0]:.6e} lr_lora={curr_lrs[1]:.6e}")
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1} finished. avg_loss={avg_loss:.4f}")
    torch.save(model.state_dict(), "lmol_model.pt")
    print("Training completed and model saved.")

if __name__ == "__main__":
    main()
