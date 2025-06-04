from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

# Truy vấn bằng ngôn ngữ tự nhiên
query = "how to read JSON file in python"
query_inputs = tokenizer(query, return_tensors="pt")
query_emb = model(**query_inputs).last_hidden_state[:, 0, :]  # CLS token

# Đoạn mã ứng viên
code_snippet = """
import json
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
"""

code_inputs = tokenizer(code_snippet, return_tensors="pt")
code_emb = model(**code_inputs).last_hidden_state[:, 0, :]

# So sánh độ tương đồng ngữ nghĩa
similarity = F.cosine_similarity(query_emb, code_emb)
print(f"🔍 Mức độ phù hợp: {similarity.item():.4f}")
