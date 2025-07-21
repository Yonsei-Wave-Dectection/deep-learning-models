import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from glob import glob

# ==== 모델 임포트 ====
from Autoformer import Autoformer
from BiLSTM import BiLSTMDenoiser
from DeNoising_Autoencoder import DenoisingAutoencoder, VariationalAutoencoder
from FFTformer import FFTformer
from Informer import Informer
from PatchTST import PatchTST
from TCN import TCNDenoiser
from WaveNet import WaveNet
from UNet1D import UNet1D

# ==== 디바이스 및 경로 설정 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ==== 데이터셋 클래스 및 전처리 ====
class SeismicDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx])

def load_and_preprocess(csv_dir, seq_len=1000):
    files = sorted(glob(os.path.join(csv_dir, '*.csv')))
    sequences = []
    for path in files:
        df = pd.read_csv(path)
        if 'amplitude' not in df.columns: continue
        amp = df['amplitude'].values
        scaler = StandardScaler()
        amp_scaled = scaler.fit_transform(amp.reshape(-1, 1)).flatten()

        # sequence split with zero-padding if needed
        for i in range(0, len(amp_scaled), seq_len):
            seq = amp_scaled[i:i+seq_len]
            if len(seq) < seq_len:
                seq = np.pad(seq, (0, seq_len - len(seq)), mode='constant')
            sequences.append(seq)

    data = np.stack(sequences)
    noise = np.random.normal(0, 0.1, data.shape)
    noisy_data = data + noise
    return noisy_data, data

def get_dataloaders(base_dir="./dataset", seq_len=1000, batch_size=16):
    train_X, train_Y = load_and_preprocess(os.path.join(base_dir, 'train'), seq_len)
    val_X, val_Y = load_and_preprocess(os.path.join(base_dir, 'val'), seq_len)
    test_X, test_Y = load_and_preprocess(os.path.join(base_dir, 'test'), seq_len)

    train_loader = DataLoader(SeismicDataset(train_X, train_Y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SeismicDataset(val_X, val_Y), batch_size=batch_size)
    test_loader = DataLoader(SeismicDataset(test_X, test_Y), batch_size=batch_size)
    return train_loader, val_loader, test_loader

# ==== 평가 함수 ====
def evaluate_scores(pred, true, threshold=0.1):
    mse = torch.mean((pred - true) ** 2).item()
    mae = torch.mean(torch.abs(pred - true)).item()

    pred_bin = (pred > threshold).int().cpu().numpy().flatten()
    true_bin = (true > threshold).int().cpu().numpy().flatten()

    acc = accuracy_score(true_bin, pred_bin)
    f1 = f1_score(true_bin, pred_bin)
    fp = ((pred_bin == 1) & (true_bin == 0)).sum()
    fn = ((pred_bin == 0) & (true_bin == 1)).sum()
    total = len(true_bin)

    return {
        "MSE": mse,
        "MAE": mae,
        "Accuracy": acc,
        "F1-score": f1,
        "False Positive Rate": fp / total,
        "False Negative Rate": fn / total,
    }

# ==== 학습 클래스 ====
class Trainer:
    def __init__(self, model, lr=1e-3):
        self.model = model.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def forward_model(self, x, y=None):
        out = self.model(x)
        pred = out[0] if isinstance(out, tuple) else out
        if y is not None and pred.shape[-1] != y.shape[-1]:
            min_len = min(pred.shape[-1], y.shape[-1])
            pred = pred[..., :min_len]
            y = y[..., :min_len]
        return pred, y

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            self.optimizer.zero_grad()
            pred, y = self.forward_model(x, y)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                pred, y = self.forward_model(x, y)
                loss = self.criterion(pred, y)
                total_loss += loss.item()
        return total_loss / len(loader)

    def test_and_evaluate(self, loader):
        self.model.eval()
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                pred, y = self.forward_model(x, y)
                return evaluate_scores(pred, y)

    def train(self, train_loader, val_loader, test_loader, model_name, epochs=5):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            print(f"[{model_name}] Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        torch.save(self.model.state_dict(), f"checkpoints/{model_name}.pt")
        scores = self.test_and_evaluate(test_loader)
        print(f"[{model_name}] Evaluation Scores: {scores}")
        return scores

# ==== 모델 리스트 정의 ====
models = {
    "Autoformer": Autoformer(),
    "BiLSTM": BiLSTMDenoiser(),
    "DenoisingAE": DenoisingAutoencoder(),
    "VAE": VariationalAutoencoder(),
    "FFTformer": FFTformer(),
    "Informer": Informer(),
    "PatchTST": PatchTST(),
    "TCN": TCNDenoiser(),
    "WaveNet": WaveNet(),
    "UNet1D": UNet1D(),
}

# ==== 실행 ====
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    results = {}

    for name, model in models.items():
        print(f"\n시작: {name}")
        trainer = Trainer(model)
        scores = trainer.train(train_loader, val_loader, test_loader, model_name=name, epochs=20)
        results[name] = scores

    df = pd.DataFrame(results).T
    df.to_csv("logs/final_evaluation_results.csv")
    print("\n전체 결과 요약:\n", df)
