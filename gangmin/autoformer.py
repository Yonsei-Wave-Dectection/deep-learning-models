import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ===== 1. 모델 클래스 정의부 (Autoformer) =====
class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.AvgPool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    def forward(self, x):
        trend = self.AvgPool(x.transpose(1,2)).transpose(1,2)
        seasonal = x - trend
        return seasonal, trend

class AutoCorrelation(nn.Module):
    def __init__(self):
        super().__init__()
    def time_delay_agg(self, values, corr):
        B, H, L, D = values.shape
        corr = F.softmax(corr, dim=-1)
        out = torch.einsum("bhld,bhld->bhld", values, corr)
        return out
    def forward(self, queries, keys, values):
        B, Lq, H, D = queries.shape
        B, Lk, H, D = keys.shape
        min_len = min(Lq, Lk)
        queries = queries[:, :min_len]
        keys = keys[:, :min_len]
        values = values[:, :min_len]
        queries_fft = torch.fft.rfft(queries, dim=1)
        keys_fft = torch.fft.rfft(keys, dim=1)
        res = queries_fft * torch.conj(keys_fft)
        corr = torch.fft.irfft(res, n=min_len, dim=1)
        out = self.time_delay_agg(values, corr)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, kernel_size):
        super().__init__()
        self.decomp = SeriesDecomposition(kernel_size)
        self.attn = AutoCorrelation()
        self.projection = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        seasonal, trend = self.decomp(x)
        res = self.attn(seasonal.unsqueeze(2), seasonal.unsqueeze(2), seasonal.unsqueeze(2)).squeeze(2)
        x = self.norm1(seasonal + self.projection(res))
        x = self.norm2(x + self.ff(x))
        return x, trend

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, kernel_size):
        super().__init__()
        self.decomp1 = SeriesDecomposition(kernel_size)
        self.decomp2 = SeriesDecomposition(kernel_size)
        self.self_attn = AutoCorrelation()
        self.cross_attn = AutoCorrelation()
        self.projection = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    def forward(self, x, cross):
        seasonal, trend1 = self.decomp1(x)
        res1 = self.self_attn(seasonal.unsqueeze(2), seasonal.unsqueeze(2), seasonal.unsqueeze(2)).squeeze(2)
        seasonal = self.norm1(seasonal + self.projection(res1))
        res2 = self.cross_attn(seasonal.unsqueeze(2), cross.unsqueeze(2), cross.unsqueeze(2)).squeeze(2)
        seasonal = self.norm2(seasonal + self.projection(res2))
        seasonal = self.norm3(seasonal + self.ff(seasonal))
        seasonal, trend2 = self.decomp2(seasonal)
        trend = trend1 + trend2
        return seasonal, trend

class Autoformer(nn.Module):
    def __init__(self, input_len, pred_len, d_model=32, e_layers=1, d_layers=1, d_ff=64, kernel_size=25):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.enc_embedding = nn.Linear(1, d_model)
        self.dec_embedding = nn.Linear(1, d_model)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, d_ff, kernel_size) for _ in range(e_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, d_ff, kernel_size) for _ in range(d_layers)])
        self.projection = nn.Linear(d_model, 1)
    def forward(self, x_enc, x_dec):
        enc_out = self.enc_embedding(x_enc)
        for layer in self.encoder:
            enc_out, _ = layer(enc_out)
        dec_out = self.dec_embedding(x_dec)
        trend = torch.zeros_like(x_dec)
        for layer in self.decoder:
            dec_out, t = layer(dec_out, enc_out)
            trend = trend + t
        dec_out = self.projection(dec_out) + trend
        return dec_out  # (batch, pred_len, 1)

# ===== 2. 데이터 불러오기 & 전처리 =====
# 파일 경로를 맞춰주세요
train_X = pd.read_csv('/content/train_X_noisy.csv', header=None).values
train_y = pd.read_csv('/content/train_y_clean.csv', header=None).values
val_X   = pd.read_csv('/content/val_X_noisy.csv', header=None).values
val_y   = pd.read_csv('/content/val_y_clean.csv', header=None).values
test_X  = pd.read_csv('/content/test_X_noisy.csv', header=None).values
test_y  = pd.read_csv('/content/test_y_clean.csv', header=None).values

# 텐서 변환 (shape: [batch, seq_len, 1])
train_X = torch.FloatTensor(train_X).unsqueeze(-1)
train_y = torch.FloatTensor(train_y).unsqueeze(-1)
val_X = torch.FloatTensor(val_X).unsqueeze(-1)
val_y = torch.FloatTensor(val_y).unsqueeze(-1)
test_X = torch.FloatTensor(test_X).unsqueeze(-1)
test_y = torch.FloatTensor(test_y).unsqueeze(-1)

# ===== 3. 학습용 배치 생성 함수 =====
def get_batch(X, Y, batch_size, shuffle=True):
    idx = torch.randperm(X.shape[0]) if shuffle else torch.arange(X.shape[0])
    for i in range(0, X.shape[0], batch_size):
        yield X[idx[i:i+batch_size]], Y[idx[i:i+batch_size]]

# ===== 4. Autoformer 학습/검증 =====
input_len = train_X.shape[1]
pred_len = train_y.shape[1]

autoformer = Autoformer(input_len=input_len, pred_len=pred_len)
optimizer = torch.optim.Adam(autoformer.parameters(), lr=1e-3)
criterion = nn.MSELoss()

train_losses, val_losses = [], []

print("Autoformer 학습 중...")
for epoch in range(50):
    autoformer.train()
    batch_losses = []
    for Xb, Yb in get_batch(train_X, train_y, batch_size=16, shuffle=True):
        X_dec = torch.zeros(Yb.shape)  # (batch, pred_len, 1)
        pred = autoformer(Xb, X_dec)
        loss = criterion(pred, Yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    train_losses.append(sum(batch_losses)/len(batch_losses))

    autoformer.eval()
    with torch.no_grad():
        X_dec_val = torch.zeros(val_y.shape)
        pred_val = autoformer(val_X, X_dec_val)
        val_loss = criterion(pred_val, val_y)
        val_losses.append(val_loss.item())

    if (epoch+1)%5 == 0:
        print(f'Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

# ===== 5. 학습/검증 Loss 시각화 =====
plt.figure(figsize=(8,4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Train/Validation Loss Curve (Overfitting Check)")
plt.legend()
plt.show()

# ===== 6. 테스트 결과 예측 및 시각화 =====
autoformer.eval()
with torch.no_grad():
    idx = 0  # 테스트 샘플 인덱스 (원하는 번호로 변경 가능)
    X_enc_test = test_X[idx:idx+1]
    X_dec_test = torch.zeros((1, pred_len, 1))
    y_true_test = test_y[idx:idx+1].squeeze(-1).numpy().flatten()
    y_pred_test = autoformer(X_enc_test, X_dec_test)[0].squeeze(-1).detach().cpu().numpy()

plt.figure(figsize=(10,4))
plt.plot(range(input_len), X_enc_test[0].squeeze(-1), label='Input(raw)', color='blue')
plt.plot(range(input_len, input_len+pred_len), y_true_test, label='Ground Truth', color='green')
plt.plot(range(input_len, input_len+pred_len), y_pred_test, label='Predicted', color='orange')
plt.title('Autoformer Prediction (raw noisy data)')
plt.legend()
plt.show()

# ===== 7. 성능 지표(MSE, MAE) 출력 =====
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(y_true_test, y_pred_test)
mae = mean_absolute_error(y_true_test, y_pred_test)
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
