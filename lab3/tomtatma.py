import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.3):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.bilstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers,
                              bidirectional=True, batch_first=True, dropout=dropout)
    
    def forward(self, x):
        # x: [batch, seq_len]
        embedded = self.embedding(x)  # [batch, seq_len, embed_size]
        outputs, (hidden, cell) = self.bilstm(embedded)  # outputs: [batch, seq_len, 2*hidden_size]
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(enc_hidden_size*2 + dec_hidden_size, dec_hidden_size)
        self.v = nn.Linear(dec_hidden_size, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        # hidden: [batch, dec_hidden_size]
        # encoder_outputs: [batch, seq_len, 2*enc_hidden_size]
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # [batch, seq_len, dec_hidden_size]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch, seq_len, dec_hidden_size]
        attention = self.v(energy).squeeze(2)  # [batch, seq_len]
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, num_layers=1, dropout=0.3):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + enc_hidden_size*2, dec_hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.attention = Attention(enc_hidden_size, dec_hidden_size)
        self.fc_out = nn.Linear(dec_hidden_size + enc_hidden_size*2 + embed_size, vocab_size)
    
    def forward(self, input, hidden, cell, encoder_outputs):
        # input: [batch] current token input ids
        input = input.unsqueeze(1)  # [batch, 1]
        embedded = self.embedding(input)  # [batch, 1, embed_size]
        
        attn_weights = self.attention(hidden[-1], encoder_outputs)  # [batch, seq_len]
        attn_weights = attn_weights.unsqueeze(1)  # [batch, 1, seq_len]
        context = torch.bmm(attn_weights, encoder_outputs)  # [batch, 1, 2*enc_hidden_size]
        
        lstm_input = torch.cat((embedded, context), dim=2)  # [batch, 1, embed+context]
        outputs, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        output = torch.cat((outputs.squeeze(1), context.squeeze(1), embedded.squeeze(1)), dim=1)  # [batch, dec_hidden+context+embed]
        prediction = self.fc_out(output)  # [batch, vocab_size]
        return prediction, hidden, cell, attn_weights.squeeze(1)

# --- Demo test ---
batch_size = 2
src_seq_len = 10
tgt_seq_len = 7
vocab_size = 1000
embed_size = 64
enc_hidden_size = 128
dec_hidden_size = 128

encoder = Encoder(vocab_size, embed_size, enc_hidden_size)
decoder = Decoder(vocab_size, embed_size, enc_hidden_size, dec_hidden_size)

src = torch.randint(0, vocab_size, (batch_size, src_seq_len))
encoder_outputs, hidden, cell = encoder(src)

# Giả sử khởi đầu decoder input là token <sos> id = 1
decoder_input = torch.LongTensor([1]*batch_size)

hidden = hidden[:decoder.lstm.num_layers]  # Lấy hidden cho decoder (chỉ 1 chiều)
cell = cell[:decoder.lstm.num_layers]

for t in range(tgt_seq_len):
    output, hidden, cell, attn_weights = decoder(decoder_input, hidden, cell, encoder_outputs)
    top1 = output.argmax(1)  # chọn token dự đoán cao nhất
    decoder_input = top1
    print(f"Step {t+1} predicted token ids:", top1.tolist())
