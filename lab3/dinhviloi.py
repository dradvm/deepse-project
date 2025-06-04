import torch
import torch.nn as nn
import torch.nn.functional as F

# Giả sử ta có 3 hàm trong dự án, mỗi hàm có vector embedding đặc trưng (ví dụ đã trích xuất trước)
# Mỗi vector biểu diễn ngữ nghĩa + cấu trúc hàm (ở đây giả lập bằng vector ngẫu nhiên)
function_embeddings = torch.tensor([
    [0.9, 0.1, 0.2],  # hàm add()
    [0.1, 0.8, 0.3],  # hàm divide()
    [0.3, 0.4, 0.7],  # hàm multiply()
])

# Vector trạng thái test case (1: pass, 0: fail)
# Giả sử có 5 test cases: test 4 pass (1), 1 fail (0)
test_results = torch.tensor([1, 1, 1, 1, 0], dtype=torch.float32)

# Mô hình đơn giản: kết hợp embedding hàm với test result để dự đoán xác suất lỗi từng hàm
class FaultLocator(nn.Module):
    def __init__(self, embed_size, test_case_size):
        super().__init__()
        self.fc1 = nn.Linear(embed_size + test_case_size, 16)
        self.fc2 = nn.Linear(16, 1)
    
    def forward(self, func_emb, test_vec):
        # Lặp qua từng hàm
        out = []
        for emb in func_emb:
            x = torch.cat([emb, test_vec])
            x = F.relu(self.fc1(x))
            x = torch.sigmoid(self.fc2(x))
            out.append(x)
        return torch.cat(out)

model = FaultLocator(embed_size=3, test_case_size=5)

# Forward pass
outputs = model(function_embeddings, test_results)
print("Xác suất lỗi từng hàm:", outputs.detach().numpy())

# Hàm có xác suất lỗi cao nhất
faulty_func_idx = torch.argmax(outputs).item()
func_names = ['add', 'divide', 'multiply']
print(f"Hàm có khả năng lỗi cao nhất: {func_names[faulty_func_idx]}")
