import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, dropout=0.3):
        super(CNNEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch, seq_len]
        x = self.embedding(x)  # [batch, seq_len, embed_dim]
        x = x.unsqueeze(1)     # [batch, 1, seq_len, embed_dim]
        conv_outputs = []
        for conv in self.convs:
            c = F.relu(conv(x)).squeeze(3)  # [batch, num_filters, seq_len - fs + 1]
            c = F.max_pool1d(c, c.size(2)).squeeze(2)  # [batch, num_filters]
            conv_outputs.append(c)
        out = torch.cat(conv_outputs, dim=1)  # [batch, num_filters * len(filter_sizes)]
        out = self.dropout(out)
        return out  # đặc trưng mã

class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(feature_dim, 1)
    
    def forward(self, features):
        # features: [batch, feature_dim]
        attn_weights = torch.sigmoid(self.attn(features))  # [batch, 1]
        weighted_features = features * attn_weights
        return weighted_features, attn_weights

class DeepJITModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_filters=100, filter_sizes=[3,4,5], dropout=0.3):
        super(DeepJITModel, self).__init__()
        self.encoder = CNNEncoder(vocab_size, embed_dim, num_filters, filter_sizes, dropout)
        self.attention = Attention(num_filters * len(filter_sizes))
        self.classifier = nn.Linear(num_filters * len(filter_sizes), 1)
    
    def forward(self, x):
        features = self.encoder(x)  # [batch, feature_dim]
        weighted_features, attn_weights = self.attention(features)
        logits = self.classifier(weighted_features).squeeze(1)  # [batch]
        return logits, attn_weights

# --- Demo sử dụng ---
batch_size = 4
seq_len = 50
vocab_size = 5000

model = DeepJITModel(vocab_size)
input_batch = torch.randint(0, vocab_size, (batch_size, seq_len))  # giả lập batch dữ liệu

logits, attn_weights = model(input_batch)
preds = torch.sigmoid(logits)

print("Output logits:", logits)
print("Predicted probabilities:", preds)
print("Attention weights:", attn_weights)
