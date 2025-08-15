# eval.py
import torch
import numpy as np
from transformers import AutoTokenizer
from datasets.scut_fbp_dataset import SCUTFBPDataset
from models.lmol_model import LMOLModel

# Load trained model
vision_model_name = "openai/clip-vit-large-patch14-336"
llm_model_name = "lmsys/vicuna-7b-v1.5"
model = LMOLModel(vision_model_name, llm_model_name)
model.load_state_dict(torch.load("lmol_model.pt"))
model.eval().cuda()

# Prepare test data and tokenizer
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
tokenizer.pad_token = tokenizer.eos_token
test_dataset = SCUTFBPDataset(split='test', download=False, transform=transform, tokenizer=tokenizer)
# Build reference set (select representative scores from training set)
reference_scores = [1.0, 2.0, 3.0, 4.0, 5.0]  # Simplified: fixed anchor score points
reference_images = []
for rs in reference_scores:
    # Select the training image closest to this score
    idx = (np.abs(train_dataset.scores - rs)).argmin()
    reference_images.append(train_dataset.image_paths[idx])
reference_tensors = [transform(Image.open(p).convert('RGB')).unsqueeze(0).cuda() for p in reference_images]
# Each element in reference_tensors: shape (1,3,336,336)

# Evaluation loop
y_true = []
y_pred = []
for i in range(len(test_dataset)):
    img1, _, _, _ = test_dataset[i]  # Get test image tensor (others unused)
    img1 = img1.unsqueeze(0).cuda()  # Add batch dimension
    # Compare against each reference image
    outcomes = []
    for ref_tensor, ref_score in zip(reference_tensors, reference_scores):
        # Use test image as first; reference image as second
        question_ids = tokenizer.encode(", which face looks more attractive?", return_tensors='pt').cuda()
        # Generate model output (assumes a forward pass providing logits)
        with torch.no_grad():
            # Set labels=None to obtain logits for custom decoding
            out = model(image1_tensor=img1, image2_tensor=ref_tensor,
                        input_ids=question_ids, labels=None)
            logits = out.logits  # shape: (1, seq_len, vocab_size)
        # Simple decoding: use last token logits to choose among "First"/"Second"/"Similar"
        # Candidate token ids
        cand_tokens = [tokenizer.encode(" First")[0], tokenizer.encode(" Second")[0], tokenizer.encode(" Similar")[0]]
        last_token_logits = logits[0, -1, :]
        # Only compare candidate token scores
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
    # Derive predicted score via BT-style heuristic based on outcomes
    # Separate wins / losses / ties
    wins = [s for (s,res) in outcomes if res == "First"]
    losses = [s for (s,res) in outcomes if res == "Second"]
    ties = [s for (s,res) in outcomes if res == "Similar"]
    if len(wins) == 0:
        pred_score = min(losses) - 0.1  # All losses: slightly below lowest anchor
    elif len(losses) == 0:
        pred_score = max(wins) + 0.1   # All wins: slightly above highest anchor
    else:
        pred_score = (max(wins) + min(losses)) / 2.0
    # Collect ground-truth and predicted scores
    y_true.append(test_dataset.scores[i])
    y_pred.append(pred_score)

# Compute PC, MAE, RMSE
y_true = np.array(y_true)
y_pred = np.array(y_pred)
pc = np.corrcoef(y_true, y_pred)[0,1]  # Pearson correlation coefficient
mae = np.mean(np.abs(y_pred - y_true))
rmse = np.sqrt(np.mean((y_pred - y_true)**2))
print(f"Evaluation results - PC: {pc:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
