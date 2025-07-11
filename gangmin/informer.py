import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

# ===== 1. Positional Encoding =====
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(1)]

# ===== 2. ProbSparse Attention =====
class ProbSparseSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, D = queries.size()
        q = self.W_q(queries).view(B, L, self.n_heads, self.d_k).transpose(1,2)
        k = self.W_k(keys).view(B, L, self.n_heads, self.d_k).transpose(1,2)
        v = self.W_v(values).view(B, L, self.n_heads, self.d_k).transpose(1,2)
        U_part = max(1, int(L * 0.5))
        q_norm = torch.norm(q, dim=-1)
        _, idx = torch.topk(q_norm, U_part, dim=-1)
        q_sparse = torch.gather(q, 2, idx.unsqueeze(-1).expand(-1,-1,-1,self.d_k))
        k_sparse = torch.gather(k, 2, idx.unsqueeze(-1).expand(-1,-1,-1,self.d_k))
        v_sparse = torch.gather(v, 2, idx.unsqueeze(-1).expand(-1,-1,-1,self.d_k))
        scores = torch.matmul(q_sparse, k_sparse.transpose(-2, -1)) / self.d_k**0.5
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v_sparse)
        zeros = torch.zeros(B, self.n_heads, L, self.d_k, device=queries.device)
        zeros.scatter_(2, idx.unsqueeze(-1).expand(-1,-1,-1,self.d_k), context)
        context = zeros
        context = context.transpose(1,2).contiguous().view(B, L, self.n_heads*self.d_k)
        out = self.out_proj(context)
        return out

# ===== 3. Encoder/Decoder with Distilling =====
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout=0.1):
        super().__init__()
        self.self_attn = ProbSparseSelfAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x2 = self.self_attn(x, x, x)
        x = x + self.dropout(x2)
        x = self.norm1(x)
        x2 = self.ff(x)
        x = x + self.dropout(x2)
        x = self.norm2(x)
        return x

class ConvDistill(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2)
    def forward(self, x):
        x = x.transpose(1,2)
        x = self.conv(x)
        x = x.transpose(1,2)
        return x

class InformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, e_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(e_layers)
        ])
        self.distill = nn.ModuleList([
            ConvDistill(d_model) for _ in range(e_layers-1)
        ])
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.distill):
                x = self.distill[i](x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout=0.1):
        super().__init__()
        self.self_attn = ProbSparseSelfAttention(d_model, n_heads, dropout)
        self.cross_attn = ProbSparseSelfAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, cross):
        x2 = self.self_attn(x, x, x)
        x = x + self.dropout(x2)
        x = self.norm1(x)
        x2 = self.cross_attn(x, cross, cross)
        x = x + self.dropout(x2)
        x = self.norm2(x)
        x2 = self.ff(x)
        x = x + self.dropout(x2)
        x = self.norm3(x)
        return x

class InformerDecoder(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, d_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(d_layers)
        ])
    def forward(self, x, cross):
        for layer in self.layers:
            x = layer(x, cross)
        return x

# ===== 4. 전체 Informer 모델 =====
class Informer(nn.Module):
    def __init__(self, input_len, pred_len, d_model=32, e_layers=2, d_layers=1, d_ff=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.enc_embedding = nn.Linear(1, d_model)
        self.dec_embedding = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.encoder = InformerEncoder(d_model, d_ff, n_heads, e_layers, dropout)
        self.decoder = InformerDecoder(d_model, d_ff, n_heads, d_layers, dropout)
        self.projection = nn.Linear(d_model, 1)
    def forward(self, x_enc, x_dec):
        x_enc = self.enc_embedding(x_enc)
        x_dec = self.dec_embedding(x_dec)
        x_enc = self.pos_enc(x_enc)
        x_dec = self.pos_enc(x_dec)
        enc_out = self.encoder(x_enc)
        dec_out = self.decoder(x_dec, enc_out)
        out = self.projection(dec_out)
        return out

# ===== 5. 데이터 불러오기 & 전처리 =====
train_X = pd.read_csv('/content/train_X_noisy.csv', header=None).values
train_y = pd.read_csv('/content/train_y_clean.csv', header=None).values
val_X   = pd.read_csv('/content/val_X_noisy.csv', header=None).values
val_y   = pd.read_csv('/content/val_y_clean.csv', header=None).values
test_X  = pd.read_csv('/content/test_X_noisy.csv', header=None).values
test_y  = pd.read_csv('/content/test_y_clean.csv', header=None).values

train_X = torch.FloatTensor(train_X).unsqueeze(-1)
train_y = torch.FloatTensor(train_y).unsqueeze(-1)
val_X   = torch.FloatTensor(val_X).unsqueeze(-1)
val_y   = torch.FloatTensor(val_y).unsqueeze(-1)
test_X  = torch.FloatTensor(test_X).unsqueeze(-1)
test_y  = torch.FloatTensor(test_y).unsqueeze(-1)

input_len = train_X.shape[1]
pred_len = train_y.shape[1]

# ===== 6. 배치 생성 함수 =====
def get_batch(X_enc, Y, batch_size, shuffle=True):
    idx = torch.randperm(X_enc.shape[0]) if shuffle else torch.arange(X_enc.shape[0])
    for i in range(0, X_enc.shape[0], batch_size):
        yield X_enc[idx[i:i+batch_size]], Y[idx[i:i+batch_size]]

# ===== 7. Informer 학습 및 검증 =====
informer = Informer(input_len=input_len, pred_len=pred_len)
optimizer = torch.optim.Adam(informer.parameters(), lr=1e-3)
criterion = nn.MSELoss()

train_losses, val_losses = [], []

print("Informer 논문 전체 구조 학습 중...")
for epoch in range(50):
    informer.train()
    batch_losses = []
    for Xb, Yb in get_batch(train_X, train_y, batch_size=16, shuffle=True):
        X_dec = torch.zeros(Yb.shape)
        pred = informer(Xb, X_dec)
        loss = criterion(pred, Yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    train_losses.append(sum(batch_losses)/len(batch_losses))

    informer.eval()
    with torch.no_grad():
        X_dec_val = torch.zeros(val_y.shape)
        pred_val = informer(val_X, X_dec_val)
        val_loss = criterion(pred_val, val_y)
        val_losses.append(val_loss.item())

    if (epoch+1)%5==0:
        print(f'Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

# ===== 8. Loss 곡선 (과적합 확인) =====
plt.figure(figsize=(8,4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Informer Train/Val Loss (Overfitting Check)")
plt.legend()
plt.show()

gap = val_losses[-1] - train_losses[-1]
print(f"최종 Train Loss: {train_losses[-1]:.4f}, 최종 Val Loss: {val_losses[-1]:.4f}, Overfit Gap: {gap:.4f}")

# ===== 9. 테스트 예측 및 시각화 =====
informer.eval()
with torch.no_grad():
    idx = 0  # 테스트 샘플 인덱스 (원하는 번호로 변경 가능)
    x_enc_test = test_X[idx:idx+1]
    x_dec_test = torch.zeros((1, pred_len, 1))
    y_true_test = test_y[idx:idx+1].squeeze(-1).numpy().flatten()
    y_pred_test = informer(x_enc_test, x_dec_test)[0].squeeze(-1).detach().cpu().numpy()

plt.figure(figsize=(10,4))
plt.plot(range(input_len), x_enc_test[0].squeeze(-1), label='Input(raw)', color='blue')
plt.plot(range(input_len, input_len+pred_len), y_true_test, label='Ground Truth', color='green')
plt.plot(range(input_len, input_len+pred_len), y_pred_test, label='Predicted', color='orange')
plt.title('Informer Prediction')
plt.legend()
plt.show()

# ===== 10. 평가 지표(MSE, MAE) =====
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(y_true_test, y_pred_test)
mae = mean_absolute_error(y_true_test, y_pred_test)
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
