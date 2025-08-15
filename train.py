# train.py
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from datasets.scut_fbp_dataset import SCUTFBPDataset
from models.lmol_model import LMOLModel
from torchvision import transforms

# 1. 參數與配置
vision_model_name = "openai/clip-vit-large-patch14-336"  # CLIP ViT-L/14 336px
llm_model_name = "lmsys/vicuna-7b-v1.5"  # Vicuna-1.5 7B (需先確保權重可用)
batch_size = 4
num_epochs = 1
learning_rate = 2e-4

# 準備 tokenizer
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
# 如果 Vicuna 使用的是 LLaMA tokenizer，可能需要 padding_side='right', truncation等
tokenizer.pad_token = tokenizer.eos_token  # 確保有定義 pad_token

# 建立數據集和 DataLoader
transform = transforms.Compose([
    transforms.Resize((336, 336)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466,0.4578275,0.40821073), std=(0.26862954,0.26130258,0.27577711))
])
train_dataset = SCUTFBPDataset(split='train', download=True, transform=transform, tokenizer=tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 2. 加載模型，應用4-bit量化和 LoRA
# 使用 BitsAndBytes 4-bit 量化配置
quant_config = BitsAndBytesConfig(load_in_4bit=True,
                                  bnb_4bit_use_double_quant=True,
                                  bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.float16)
# 加載 LMOL 模型，其中包含 CLIP 和 Vicuna
device_map = "auto"  # 自動將模型分配到 GPU
lmol_model = LMOLModel(vision_model_name, llm_model_name)
lmol_model = lmol_model.to('cuda')  # 移動到 GPU

# 構造 LoRA 配置
lora_config = LoraConfig(
    r=8, lora_alpha=32, target_modules=["q_proj","v_proj","k_proj","o_proj"], # 針對 LLaMA注意力和輸出層
    lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
)
# 將 LoRA 模型應用到 LMOL 模型的語言部分
lmol_model.language_model = get_peft_model(lmol_model.language_model, lora_config)
lmol_model.train()
print("Trainable parameters:")
lmol_model.language_model.print_trainable_parameters()  # 列出可訓練參數（應該主要是LoRA層和投影矩陣）

# 凍結不需訓練的參數（如CLIP已在init中凍結, Vicuna量化後的基礎權重在PEFT中預設不計算梯度）
for param in lmol_model.visual_encoder.parameters():
    param.requires_grad = False
for name, param in lmol_model.language_model.named_parameters():
    # 確保只訓練LoRA相關參數以及投影矩陣
    if "lora" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
for param in lmol_model.projection.parameters():
    param.requires_grad = True

# 構造優化器（只包含LoRA和投影矩陣的參數）
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, lmol_model.parameters()), lr=learning_rate)

# 3. 訓練循環
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in train_loader:
        img1, img2, input_ids, labels = batch
        img1 = img1.cuda(); img2 = img2.cuda()
        input_ids = input_ids.cuda(); labels = labels.cuda()
        optimizer.zero_grad()
        outputs = lmol_model(image1_tensor=img1, image2_tensor=img2,
                              input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} finished, average loss: {avg_loss:.4f}")
# 訓練完成後，可將模型保存
torch.save(lmol_model.state_dict(), "lmol_model.pt")
print("Training completed and model saved.")
