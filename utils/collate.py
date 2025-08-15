# utils/collate.py
# -*- coding: utf-8 -*-

from typing import List, Tuple, Dict, Any
import torch
from torch.nn.utils.rnn import pad_sequence

def lmol_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Dict[str, Any]:
    """
    Collate function for LMOL pairs.
    Each item in 'batch' is a tuple: (img1, img2, input_ids, labels)
    - img1, img2: FloatTensor [3, H, W]
    - input_ids: LongTensor [T_i]
    - labels: LongTensor [T_i] (prompt tokens should already be -100 where needed)

    This function:
      * stacks images as [B, 3, H, W]
      * pads input_ids and labels to the same length across batch
      * builds attention_mask aligned to input_ids
    """
    imgs1 = [b[0] for b in batch]
    imgs2 = [b[1] for b in batch]
    input_ids = [b[2] for b in batch]
    labels = [b[3] for b in batch]

    # Stack images
    img1_batch = torch.stack(imgs1, dim=0)  # [B, 3, H, W]
    img2_batch = torch.stack(imgs2, dim=0)  # [B, 3, H, W]

    # Pad sequences
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = (input_ids_padded != 0).long()

    return {
        "img1": img1_batch,
        "img2": img2_batch,
        "input_ids": input_ids_padded,
        "labels": labels_padded,
        "attention_mask": attention_mask,
    }
