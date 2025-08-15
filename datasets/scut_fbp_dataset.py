import os
import zipfile
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class SCUTFBPDataset(Dataset):
    """
    SCUT-FBP5500 數據集，支持離線平衡配對與可配置的 theta、M。
    """

    def __init__(
        self,
        root_dir: str = "data/SCUT-FBP5500",
        split: str = "train",
        download: bool = True,
        transform=None,
        tokenizer=None,
        *,
        theta: float = 0.2,
        M: int = 30000,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        self.theta = theta
        self.max_pairs_per_relation = M

        if download:
            self._download_dataset()

        img_dir = os.path.join(self.root_dir, "Images")
        split_file = os.path.join(self.root_dir, f"{split}_test_split.txt")
        self.image_paths: List[str] = []
        scores: List[float] = []
        with open(split_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                rel_path, score = parts[0], float(parts[1])
                self.image_paths.append(os.path.join(img_dir, rel_path))
                scores.append(score)
        self.scores = np.array(scores)
        self.num_images = len(self.image_paths)

        # 建立 balanced pairs
        self.pairs: List[Tuple[int, int, str]] = []
        pairs_first: List[Tuple[int, int, str]] = []
        pairs_second: List[Tuple[int, int, str]] = []
        pairs_similar: List[Tuple[int, int, str]] = []

        n = self.num_images
        for i in range(n):
            if (
                len(pairs_first) >= self.max_pairs_per_relation
                and len(pairs_second) >= self.max_pairs_per_relation
                and len(pairs_similar) >= self.max_pairs_per_relation
            ):
                break
            for j in range(i + 1, n):
                if (
                    len(pairs_first) >= self.max_pairs_per_relation
                    and len(pairs_second) >= self.max_pairs_per_relation
                    and len(pairs_similar) >= self.max_pairs_per_relation
                ):
                    break
                yi, yj = self.scores[i], self.scores[j]
                diff = yi - yj
                if abs(diff) <= self.theta:
                    if len(pairs_similar) < self.max_pairs_per_relation:
                        pairs_similar.append((i, j, "Similar."))
                elif diff > self.theta:
                    if len(pairs_first) < self.max_pairs_per_relation:
                        pairs_first.append((i, j, "First."))
                else:
                    if len(pairs_second) < self.max_pairs_per_relation:
                        pairs_second.append((i, j, "Second."))
        min_count = min(len(pairs_first), len(pairs_second), len(pairs_similar))
        if min_count == 0:
            raise RuntimeError("Not enough pairs collected; adjust M or theta.")
        pairs_first = pairs_first[:min_count]
        pairs_second = pairs_second[:min_count]
        pairs_similar = pairs_similar[:min_count]
        self.pairs = pairs_first + pairs_second + pairs_similar
        np.random.shuffle(self.pairs)
        self.num_pairs = len(self.pairs)

    def _download_dataset(self) -> None:
        os.makedirs(self.root_dir, exist_ok=True)
        archive_path = os.path.join(self.root_dir, "SCUT-FBP5500.zip")
        if not os.path.exists(archive_path):
            print("Downloading SCUT-FBP5500 dataset...")
            url = "https://drive.google.com/uc?export=download&id=1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf"
            import requests
            response = requests.get(url)
            with open(archive_path, "wb") as f:
                f.write(response.content)
            print("Download complete. Extracting...")
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(self.root_dir)
            print("Extraction finished.")

    def __len__(self) -> int:
        return self.num_pairs

    def __getitem__(self, index: int):
        idx1, idx2, label_text = self.pairs[index]
        img1 = Image.open(self.image_paths[idx1]).convert("RGB")
        img2 = Image.open(self.image_paths[idx2]).convert("RGB")
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        instruction = ", which face looks more attractive?"
        prompt_ids = self.tokenizer.encode(instruction, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(" " + label_text, add_special_tokens=False)
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return img1, img2, input_ids, labels
