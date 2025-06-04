import ast
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Hàm chuyển mã nguồn thành chuỗi loại node AST
def ast_to_sequence(code_str):
    tree = ast.parse(code_str)
    nodes = []
    def visit(node):
        nodes.append(type(node).__name__)
        for child in ast.iter_child_nodes(node):
            visit(child)
    visit(tree)
    return nodes

# 2. Tạo từ điển node type
def build_vocab(sequences):
    vocab = {"<PAD>":0, "<UNK>":1}
    idx = 2
    for seq in sequences:
        for token in seq:
            if token not in vocab:
                vocab[token] = idx
                idx +=1
    return vocab

# 3. Mã hóa chuỗi node thành ID
def encode_sequence(seq, vocab, max_len=50):
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in seq]
    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

# 4. Mạng LSTM đơn giản để học embedding AST sequence
class ASTEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(ASTEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
    
    def forward(self, x):
        emb = self.embedding(x)
        outputs, (hidden, _) = self.lstm(emb)
        return hidden[-1]  # Lấy hidden cuối cùng làm embedding đoạn mã

# 5. Hàm so sánh vector embedding (cosine similarity)
def similarity(vec1, vec2):
    vec1 = F.normalize(vec1, dim=0)
    vec2 = F.normalize(vec2, dim=0)
    return torch.dot(vec1, vec2).item()

# --- Demo sử dụng ---
code1 = """
sum = 0
for i in range(len(array)):
    sum += array[i]
"""

code2 = """
sum = 0
i = 0
while i < len(array):
    sum = sum + array[i]
    i += 1
"""

seqs = [ast_to_sequence(code1), ast_to_sequence(code2)]
vocab = build_vocab(seqs)

seq1_ids = torch.tensor([encode_sequence(seqs[0], vocab)])
seq2_ids = torch.tensor([encode_sequence(seqs[1], vocab)])

model = ASTEncoder(len(vocab), embed_size=32, hidden_size=64)

vec1 = model(seq1_ids).squeeze(0)
vec2 = model(seq2_ids).squeeze(0)

sim = similarity(vec1, vec2)
print(f"Cosine similarity between code snippets: {sim:.4f}")

if sim > 0.8:
    print("Hai đoạn mã được coi là sao chép ngữ nghĩa (semantic clones).")
else:
    print("Hai đoạn mã không phải sao chép ngữ nghĩa.")
