import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ===== 1. 가짜 지진파 생성 =====
def generate_fake_seismic_signal(length=1000, noise_level=0.2):
    t = np.linspace(0, 20*np.pi, length)
    wave = 0.3 * np.sin(t)
    quake = np.zeros_like(wave)
    quake[400:500] = np.sin(np.linspace(0, 15*np.pi, 100)) * 2.0
    noise = noise_level * np.random.randn(length)
    return wave + quake + noise

# ===== 2. Noise2Noise Denoiser (1D-CNN) =====
class N2NDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=5, padding=2)
        )
    def forward(self, x):
        return self.net(x)

def make_n2n_batch(raw_signal, noise_level=0.2, batch=32, length=512):
    X1, X2 = [], []
    for _ in range(batch):
        noise1 = noise_level * np.random.randn(length)
        noise2 = noise_level * np.random.randn(length)
        x1 = raw_signal[:length] + noise1
        x2 = raw_signal[:length] + noise2
        X1.append(x1)
        X2.append(x2)
    X1 = torch.FloatTensor(X1).unsqueeze(1)
    X2 = torch.FloatTensor(X2).unsqueeze(1)
    return X1, X2

# ===== 3. PatchTST =====
class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, stride, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.proj = nn.Linear(patch_len, d_model)
    def forward(self, x):
        B, L, _ = x.shape
        patches = []
        for i in range(0, L - self.patch_len + 1, self.stride):
            patch = x[:, i:i+self.patch_len, 0]
            patches.append(patch.unsqueeze(1))
        patches = torch.cat(patches, dim=1)
        out = self.proj(patches)
        return out

class PatchTST(nn.Module):
    def __init__(self, input_len, pred_len, patch_len=16, stride=8, d_model=64, n_layers=2, n_heads=4, d_ff=128, dropout=0.1):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.embed = PatchEmbedding(patch_len, stride, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(((input_len - patch_len)//stride + 1)*d_model, pred_len)
        )
    def forward(self, x):
        x_embed = self.embed(x)
        enc_out = self.encoder(x_embed)
        out = self.head(enc_out)
        return out.unsqueeze(-1)

# ===== 4. 학습/실험 파이프라인 =====
input_len = 96
pred_len = 48

raw_signal = generate_fake_seismic_signal(length=512, noise_level=0) #전처리 데이터 받으면 여기에!
n2n_model = N2NDenoiser()
n2n_optimizer = torch.optim.Adam(n2n_model.parameters(), lr=1e-3)
n2n_criterion = nn.MSELoss()

print("Noise2Noise 학습 중...")
for epoch in range(500):
    X1, X2 = make_n2n_batch(raw_signal, batch=32, length=512)
    output = n2n_model(X1)
    loss = n2n_criterion(output, X2)
    n2n_optimizer.zero_grad()
    loss.backward()
    n2n_optimizer.step()

with torch.no_grad():
    test_noisy = raw_signal + 0.2*np.random.randn(512)
    test_noisy_tensor = torch.FloatTensor(test_noisy).unsqueeze(0).unsqueeze(0)
    denoised = n2n_model(test_noisy_tensor).squeeze().numpy()

def make_patchtst_batch(clean_signal, denoised_signal, input_len, pred_len, batch=16):
    X, Y = [], []
    length = len(clean_signal)
    for i in range(batch):
        start = np.random.randint(0, length - input_len - pred_len)
        x = denoised_signal[start:start+input_len]
        y = clean_signal[start+input_len:start+input_len+pred_len]
        X.append(x)
        Y.append(y)
    X = torch.FloatTensor(X).unsqueeze(-1)
    Y = torch.FloatTensor(Y).unsqueeze(-1)
    return X, Y

patchtst = PatchTST(input_len=input_len, pred_len=pred_len)
optimizer = torch.optim.Adam(patchtst.parameters(), lr=1e-3)
criterion = nn.MSELoss()

print("PatchTST 학습 중...")
for epoch in range(500):
    X, Y = make_patchtst_batch(raw_signal, denoised, input_len, pred_len, batch=16)
    pred = patchtst(X)
    loss = criterion(pred, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    idx = np.random.randint(0, len(raw_signal)-input_len-pred_len)
    x_test = denoised[idx:idx+input_len]
    y_true = raw_signal[idx+input_len:idx+input_len+pred_len]
    x_test = torch.FloatTensor(x_test).unsqueeze(0).unsqueeze(-1)
    y_pred = patchtst(x_test)[0].squeeze(-1).detach().cpu().numpy()

plt.figure(figsize=(10,4))
plt.plot(range(input_len), x_test[0].squeeze(-1), label='Input(denoised)', color='blue')
plt.plot(range(input_len, input_len+pred_len), y_true, label='Ground Truth', color='green')
plt.plot(range(input_len, input_len+pred_len), y_pred, label='Predicted', color='orange')
plt.title('PatchTST Prediction (N2N denoising + PatchTST)')
plt.legend()
plt.show()

# ===== 6. 정확성(MSE, MAE) 계산 =====
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
