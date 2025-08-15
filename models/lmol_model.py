# models/lmol_model.py
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, AutoModelForCausalLM

class LMOLModel(nn.Module):
    def __init__(self, vision_model_name: str, llm_model_name: str):
        super().__init__()
        # 1. 視覺編碼器：載入預訓練的 CLIP ViT-L/14-336px 並凍結參數
        self.visual_encoder = CLIPVisionModel.from_pretrained(
            vision_model_name
        )  # 輸出 shape: (batch, num_patches, vision_hidden_dim)
        self.visual_encoder.eval()
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

        vision_hidden_dim = self.visual_encoder.config.hidden_size  # CLIP視覺特徵維度
        # 2. 投影矩陣：將視覺特徵映射到語言模型的embedding維度
        # 使用兩層帶非線性激活的 MLP（LLaVA-1.5採用兩層MLP:contentReference[oaicite:14]{index=14}）
        llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        language_hidden_dim = llm.config.hidden_size  # LLM隱層維度 (Vicuna-1.5)
        self.projection = nn.Sequential(
            nn.Linear(vision_hidden_dim, language_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(language_hidden_dim, language_hidden_dim),
        )
        # 3. LLM比較器：Vicuna 作為基礎的自回歸模型
        self.language_model = llm  # Vicuna-1.5 (需相容 HuggingFace transformers)

    def forward(self, image1_tensor, image2_tensor, input_ids, attention_mask=None, labels=None):
        """
        前向傳播：將兩張圖像與文本指令一起送入模型，輸出Loss（訓練時）或Logits（推論時）。
        image1_tensor, image2_tensor: [batch, 3, H, W] 的圖像張量（像素已標準化）
        input_ids: [batch, T_text] 文本輸入的 token IDs（提問 + 可能的答案序列）
        labels: [batch, T_text] 文本標籤，用於計算訓練 loss（非訓練階段可為 None）
        """
        bsz = input_ids.size(0)
        device = input_ids.device

        # 使用視覺編碼器編碼圖像，獲取每張圖像的 patch 特徵序列
        with torch.no_grad():
            img1_feats = self.visual_encoder(image1_tensor).last_hidden_state  # shape: (bsz, N_patch1, vis_hidden)
            img2_feats = self.visual_encoder(image2_tensor).last_hidden_state  # shape: (bsz, N_patch2, vis_hidden)
        # 投影到語言模型的嵌入空間
        img1_tokens = self.projection(img1_feats)  # shape: (bsz, N_patch1, lang_hidden)
        img2_tokens = self.projection(img2_feats)  # shape: (bsz, N_patch2, lang_hidden)
        # 將兩個圖像的token序列按順序維度拼接
        visual_tokens = torch.cat([img1_tokens, img2_tokens], dim=1)  # (bsz, N_patch1+N_patch2, lang_hidden)
        visual_token_count = visual_tokens.size(1)

        # 取得LLM的詞嵌入層，將input_ids轉換為對應的text embedding序列
        text_embeds = self.language_model.model.get_input_embeddings()(input_ids)  # (bsz, T_text, lang_hidden)
        # 將圖像token和文本token的embedding序列拼接為完整輸入
        inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)  # (bsz, N_total+T_text, lang_hidden)
        # 生成對應的注意力mask：圖像token部分全為1，後接文本的attention mask
        if attention_mask is not None:
            # 構造新mask，前面的圖像tokens全部置1
            ones = torch.ones((bsz, visual_token_count), dtype=torch.long, device=device)
            combined_mask = torch.cat([ones, attention_mask], dim=1)
        else:
            combined_mask = None

        # 語言模型前向計算（embedding輸入）
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            labels=labels  # 若提供 labels，模型會計算 cross-entropy loss
        )
        return outputs  # outputs.loss (若提供labels) 以及 outputs.logits 等
