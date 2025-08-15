# eval.py
import torch
import numpy as np
from transformers import AutoTokenizer
from datasets.scut_fbp_dataset import SCUTFBPDataset
from models.lmol_model import LMOLModel

# 加載訓練好的模型
vision_model_name = "openai/clip-vit-large-patch14-336"
llm_model_name = "lmsys/vicuna-7b-v1.5"
model = LMOLModel(vision_model_name, llm_model_name)
model.load_state_dict(torch.load("lmol_model.pt"))
model.eval().cuda()

# 準備測試資料和 tokenizer
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
tokenizer.pad_token = tokenizer.eos_token
test_dataset = SCUTFBPDataset(split='test', download=False, transform=transform, tokenizer=tokenizer)
# 構建參考集 (比如從訓練集中挑選若干具有代表性的分數作為參考)
reference_scores = [1.0, 2.0, 3.0, 4.0, 5.0]  # 簡化起見，以固定分數點作參考
reference_images = []
for rs in reference_scores:
    # 選取最接近該分數的訓練圖像
    idx = (np.abs(train_dataset.scores - rs)).argmin()
    reference_images.append(train_dataset.image_paths[idx])
reference_tensors = [transform(Image.open(p).convert('RGB')).unsqueeze(0).cuda() for p in reference_images]
# reference_tensors 列表中每個元素: shape (1,3,336,336)

# 評估循環
y_true = []
y_pred = []
for i in range(len(test_dataset)):
    img1, _, _, _ = test_dataset[i]  # 獲取測試圖像張量（img2, input_ids, labels此處無用）
    img1 = img1.unsqueeze(0).cuda()  # 添加batch維度
    # 對每個參考圖像進行比較
    outcomes = []
    for ref_tensor, ref_score in zip(reference_tensors, reference_scores):
        # 將測試圖像(img1)作為第一張，參考圖像作為第二張，組成輸入
        question_ids = tokenizer.encode(", which face looks more attractive?", return_tensors='pt').cuda()
        # **生成模型輸出**：推理時我們需要生成回答，這裡簡化假設模型有一個 infer 方法
        with torch.no_grad():
            # 將 labels 設為 None，以獲取 logits 進行自定義解碼
            out = model(image1_tensor=img1, image2_tensor=ref_tensor,
                        input_ids=question_ids, labels=None)
            logits = out.logits  # shape: (1, seq_len, vocab_size)
        # 簡單解碼：取最後一個token的logits，找到對應 "First"/"Second"/"Similar" 幾個詞概率最高者
        # 獲取候選詞的token ids
        cand_tokens = [tokenizer.encode(" First")[0], tokenizer.encode(" Second")[0], tokenizer.encode(" Similar")[0]]
        last_token_logits = logits[0, -1, :]
        # 只比較候選token的分數
        cand_logits = last_token_logits[cand_tokens]
        choice = torch.argmax(cand_logits).item()
        answer_token_id = cand_tokens[choice]
        answer_text = tokenizer.decode([answer_token_id]).strip()
        if answer_text == "First":
            outcomes.append((ref_score, "First"))
        elif answer_text == "Second":
            outcomes.append((ref_score, "Second"))
        else:
            outcomes.append((ref_score, "Similar"))
    # 根據 outcomes 應用 BT 模型計算預測分數
    # 提取所有 ref_score，以及哪些贏/輸
    wins = [s for (s,res) in outcomes if res == "First"]
    losses = [s for (s,res) in outcomes if res == "Second"]
    ties = [s for (s,res) in outcomes if res == "Similar"]
    if len(wins) == 0:
        pred_score = min(losses) - 0.1  # 若全部敗，預測略低於最低參考
    elif len(losses) == 0:
        pred_score = max(wins) + 0.1   # 若全部勝，預測略高於最高參考
    else:
        pred_score = (max(wins) + min(losses)) / 2.0
    # 收集真實值與預測值
    y_true.append(test_dataset.scores[i])
    y_pred.append(pred_score)

# 計算 PC, MAE, RMSE
y_true = np.array(y_true)
y_pred = np.array(y_pred)
pc = np.corrcoef(y_true, y_pred)[0,1]  # 皮爾森相關係數
mae = np.mean(np.abs(y_pred - y_true))
rmse = np.sqrt(np.mean((y_pred - y_true)**2))
print(f"Evaluation results - PC: {pc:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
