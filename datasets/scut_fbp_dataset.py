# datasets/scut_fbp_dataset.py
import os, zipfile, requests
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class SCUTFBPDataset(Dataset):
    def __init__(self, root_dir='data/SCUT-FBP5500', split='train', download=True, transform=None, tokenizer=None):
        """
        初始化 SCUT-FBP5500 數據集。
        - 若本地不存在資料且 download=True，會自動從官方提供的鏈接下載並解壓資料集。
        - 將原始標註轉換為 LMOL 所需的指令格式資料，包括成對的圖像及對應文本提示和答案。
        """
        self.root_dir = root_dir
        self.split = split  # 'train' 或 'test'
        self.transform = transform  # 圖像預處理方法 (調整大小、歸一化等)
        self.tokenizer = tokenizer  # 文本tokenizer，用於將指令和答案轉為input_ids

        # 資料下載與解壓
        if download:
            self._download_dataset()

        # 讀取圖像路徑與對應評分
        img_dir = os.path.join(self.root_dir, 'Images')
        anno_file = os.path.join(self.root_dir, f'{split}_test_split.txt')  # 假設有提供訓練/測試的檔案列表
        self.image_paths = []
        self.scores = []
        with open(anno_file, 'r') as f:
            for line in f:
                path, score = line.strip().split()
                # e.g., line format: "AF/AF134.jpg 3.75"
                self.image_paths.append(os.path.join(img_dir, path))
                self.scores.append(float(score))
        self.scores = np.array(self.scores)
        self.num_images = len(self.image_paths)

        # 生成平衡的序對 (balanced ordinal pairs) 列表
        self.pairs = []  # list of tuples: (idx1, idx2, answer_text)
        # 閾值θ=0.2，用於判定 "≈" 類別:contentReference[oaicite:24]{index=24}
        theta = 0.2
        # 我們將從三類序關係中各抽取 M 對，以構造平衡的訓練資料集:contentReference[oaicite:25]{index=25}。
        M = 30000  # 每類選擇對數 (總計約 90k 對，LMOL 論文中使用每fold 90k 對作訓練:contentReference[oaicite:26]{index=26})
        pairs_similar = []
        pairs_first = []
        pairs_second = []

        # 雙重迴圈掃描所有圖像對（注意：5500^2 略大，此處簡化處理以構造平衡樣本集）
        n = self.num_images
        for i in range(n):
            if len(pairs_first) >= M and len(pairs_second) >= M and len(pairs_similar) >= M:
                break  # 各類別已滿
            for j in range(i+1, n):
                if len(pairs_first) >= M and len(pairs_second) >= M and len(pairs_similar) >= M:
                    break
                yi, yj = self.scores[i], self.scores[j]
                diff = yi - yj
                if abs(diff) <= theta:
                    # 標記為 "Similar."
                    if len(pairs_similar) < M:
                        pairs_similar.append((i, j, "Similar."))
                elif diff > theta:
                    # i比j美 -> 第一張更美 => "First."
                    if len(pairs_first) < M:
                        pairs_first.append((i, j, "First."))
                elif diff < -theta:
                    # i比j醜 -> 第二張更美 => 輸出"Second."
                    if len(pairs_second) < M:
                        pairs_second.append((i, j, "Second."))
            # end for j
        # end for i

        # 平衡取樣：確保三種關係數量大致相等:contentReference[oaicite:27]{index=27}
        # 若某類不足M，允許重複取樣或減少M以平衡；這裡採用等量取最小長度
        min_count = min(len(pairs_first), len(pairs_second), len(pairs_similar))
        if min_count == 0:
            raise RuntimeError("Not enough pairs collected. Consider adjusting M or theta.")
        pairs_first = pairs_first[:min_count]
        pairs_second = pairs_second[:min_count]
        pairs_similar = pairs_similar[:min_count]
        self.pairs = pairs_first + pairs_second + pairs_similar

        # Shuffle 所有 pair
        np.random.shuffle(self.pairs)
        self.num_pairs = len(self.pairs)

    def _download_dataset(self):
        # 若資料夾不存在則下載並解壓
        os.makedirs(self.root_dir, exist_ok=True)
        img_archive = os.path.join(self.root_dir, 'SCUT-FBP5500.zip')
        if not os.path.exists(img_archive):
            print("Downloading SCUT-FBP5500 dataset...")
            url = "https://drive.google.com/uc?export=download&id=1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf"
            r = requests.get(url)
            with open(img_archive, 'wb') as f:
                f.write(r.content)
            print("Download complete. Extracting files...")
            with zipfile.ZipFile(img_archive, 'r') as zip_ref:
                zip_ref.extractall(self.root_dir)
            print("Extraction done.")

    def __len__(self):
        # 返回成對樣本總數
        return self.num_pairs

    def __getitem__(self, index):
        # 根據索引返回一組pair樣本，包括兩張圖像的tensor、文本提示的input_ids、label等
        idx1, idx2, answer_text = self.pairs[index]
        img_path1 = self.image_paths[idx1]
        img_path2 = self.image_paths[idx2]
        # 讀取並轉換圖像
        img1 = Image.open(img_path1).convert('RGB')
        img2 = Image.open(img_path2).convert('RGB')
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        # 構建文本指令: 兩張圖像後緊跟問句
        instruction = ", which face looks more attractive?"
        # 利用 tokenizer 將 prompt（問句部分）和答案拼接，並產生 labels（將問句部分的 label 設為忽略）
        # 模型最終只需要預測答案部分:contentReference[oaicite:28]{index=28}。
        # 我們期望模型輸出 answer_text，例如 "First."
        prompt_ids = self.tokenizer.encode(instruction, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(" " + answer_text, add_special_tokens=False)
        # 這裡在答案前加一個空格，確保tokenizer將"First."等單詞作為單獨token編碼
        input_ids = prompt_ids + answer_ids
        # 構造 labels，將 prompt 部分標記為 -100（忽略），答案部分與 input_ids 相同
        labels = [-100] * len(prompt_ids) + answer_ids
        # 注意：在 Vicuna/LLama 模型中通常會在序列開頭附加 BOS (<s>) token；此處簡化處理
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return img1, img2, input_ids, labels
