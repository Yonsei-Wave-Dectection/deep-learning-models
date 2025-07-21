import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

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
from common_data_processing import SeismicDataset, SeismicDataLoader

# ==== 디바이스 및 경로 설정 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

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

# ==== 모델 학습 클래스 ====
class Trainer:
    def __init__(self, model, lr=1e-3):
        self.model = model.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def forward_model(self, x):
        out = self.model(x)
        return out[0], out[1] if isinstance(out, tuple) else (out, torch.zeros_like(out))

    def train_epoch(self, loader):
        self.model.train()
        total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            self.optimizer.zero_grad()
            pred, _ = self.forward_model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()
            total += loss.item()
        return total / len(loader)

    def validate(self, loader):
        self.model.eval()
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                pred, _ = self.forward_model(x)
                loss = self.criterion(pred, y)
                total += loss.item()
        return total / len(loader)

    def test_and_evaluate(self, loader):
        self.model.eval()
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                pred, _ = self.forward_model(x)
                return evaluate_scores(pred, y)

    def train(self, train_loader, val_loader, test_loader, model_name, epochs=5):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            print(f"[{model_name}] Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save model
        torch.save(self.model.state_dict(), f"checkpoints/{model_name}.pt")

        # Evaluate
        scores = self.test_and_evaluate(test_loader)
        print(f"[{model_name}] Evaluation Scores: {scores}")
        return scores

# ==== 데이터 로딩 ====
def load_datasets(base_dir="./dataset", seq_len=1000, batch_size=16):
    loaders = {}
    for split in ["train", "val", "test"]:
        loader = SeismicDataLoader(csv_dir=os.path.join(base_dir, split), sequence_length=seq_len)
        raw_data, _ = loader.load_csv_files()
        sequences = loader.create_sequences(raw_data)
        noisy, clean = loader.add_synthetic_noise(sequences)
        dataset = SeismicDataset(noisy, clean, sequence_length=seq_len)
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))
    return loaders["train"], loaders["val"], loaders["test"]

# ==== 모델 리스트 정의 ====

### 님들은 이 부분만 수정하면 됨. 
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

# ==== 전체 실행 ====
if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_datasets()
    results = {}

    for name, model in models.items():
        print(f"\n시작: {name}")
        trainer = Trainer(model)
        scores = trainer.train(train_loader, val_loader, test_loader, model_name=name, epochs=5)
        results[name] = scores

    # 전체 결과 저장
    df = pd.DataFrame(results).T
    df.to_csv("logs/final_evaluation_results.csv")
    print("\n전체 결과 요약:\n", df)

