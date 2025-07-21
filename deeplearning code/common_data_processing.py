import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

class SeismicDataset(Dataset):
    """지진파 데이터셋 클래스"""
    def __init__(self, data: np.ndarray, labels: np.ndarray, sequence_length: int = 1000):
        self.data = data
        self.labels = labels
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx])
        y = torch.FloatTensor(self.labels[idx])
        return x, y

class SeismicDataLoader:
    """지진파 데이터 로더 및 전처리 클래스"""
    
    def __init__(self, csv_dir: str, sequence_length: int = 1000, overlap_ratio: float = 0.5):
        self.csv_dir = csv_dir
        self.sequence_length = sequence_length
        self.overlap_ratio = overlap_ratio
        self.scaler = StandardScaler()
        
    def load_csv_files(self) -> Tuple[np.ndarray, List[str]]:
        """CSV 파일들을 로드하고 시계열 데이터로 변환"""
        all_data = []
        file_names = []
        
        csv_files = glob.glob(os.path.join(self.csv_dir, "*.csv"))
        
        for file_path in sorted(csv_files):
            try:
                df = pd.read_csv(file_path)
                # 시간 컬럼 제거하고 진폭 데이터만 사용
                amplitude_data = df['amplitude'].values
                all_data.append(amplitude_data)
                file_names.append(os.path.basename(file_path))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        return np.array(all_data, dtype=object), file_names
    
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """시계열 데이터를 고정 길이 시퀀스로 분할"""
        sequences = []
        
        for signal in data:
            signal_length = len(signal)
            step_size = int(self.sequence_length * (1 - self.overlap_ratio))
            
            for start_idx in range(0, signal_length - self.sequence_length + 1, step_size):
                end_idx = start_idx + self.sequence_length
                sequence = signal[start_idx:end_idx]
                sequences.append(sequence)
                
        return np.array(sequences)
    
    def add_synthetic_noise(self, clean_data: np.ndarray, noise_factor: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """합성 노이즈를 추가하여 학습 데이터 생성"""
        # 깨끗한 신호를 타겟으로 사용
        clean_signals = clean_data.copy()
        
        # 노이즈가 추가된 신호를 입력으로 사용
        noisy_signals = clean_data.copy()
        
        for i in range(len(noisy_signals)):
            # 가우시안 노이즈 추가
            gaussian_noise = np.random.normal(0, noise_factor * np.std(clean_data[i]), clean_data[i].shape)
            
            # 임펄스 노이즈 추가 (10% 확률)
            impulse_mask = np.random.random(clean_data[i].shape) < 0.1
            impulse_noise = np.random.normal(0, 3 * noise_factor * np.std(clean_data[i]), clean_data[i].shape)
            
            noisy_signals[i] = clean_data[i] + gaussian_noise + impulse_mask * impulse_noise
            
        return noisy_signals, clean_signals
    
    def prepare_data(self, test_size: float = 0.2, validation_size: float = 0.1, noise_factor: float = 0.1):
        """전체 데이터 준비 파이프라인"""
        print("Loading CSV files...")
        raw_data, file_names = self.load_csv_files()
        
        print("Creating sequences...")
        sequences = self.create_sequences(raw_data)
        
        print("Adding synthetic noise...")
        noisy_data, clean_data = self.add_synthetic_noise(sequences, noise_factor)
        
        print("Normalizing data...")
        # 노이즈가 있는 데이터로 스케일러 학습
        noisy_flat = noisy_data.reshape(-1, 1)
        self.scaler.fit(noisy_flat)
        
        # 정규화 적용
        noisy_normalized = np.array([
            self.scaler.transform(seq.reshape(-1, 1)).flatten() 
            for seq in noisy_data
        ])
        
        clean_normalized = np.array([
            self.scaler.transform(seq.reshape(-1, 1)).flatten() 
            for seq in clean_data
        ])
        
        # 데이터 분할
        X_temp, X_test, y_temp, y_test = train_test_split(
            noisy_normalized, clean_normalized, test_size=test_size, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_size/(1-test_size), random_state=42
        )
        
        print(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'scaler': self.scaler,
            'file_names': file_names
        }

class UncertaintyQuantifier:
    """불확실성 정량화를 위한 유틸리티 클래스"""
    
    @staticmethod
    def monte_carlo_dropout(model, x, n_samples=100, dropout_p=0.1):
        """Monte Carlo Dropout을 사용한 불확실성 추정"""
        model.eval()  # 평가 모드로 설정하지만 dropout은 활성화
        predictions = []
        
        # Dropout을 활성화하기 위해 train 모드로 전환
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = model(x)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
    
    @staticmethod
    def ensemble_uncertainty(models, x):
        """앙상블을 사용한 불확실성 추정"""
        predictions = []
        
        for model in models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred

def calculate_metrics(y_true, y_pred, uncertainty=None):
    """모델 성능 메트릭 계산"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Signal-to-Noise Ratio
    signal_power = np.mean(y_true ** 2)
    noise_power = np.mean((y_true - y_pred) ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'SNR': snr
    }
    
    if uncertainty is not None:
        # 불확실성 관련 메트릭
        mean_uncertainty = np.mean(uncertainty)
        uncertainty_correlation = np.corrcoef(uncertainty.flatten(), 
                                            np.abs(y_true - y_pred).flatten())[0, 1]
        metrics.update({
            'Mean_Uncertainty': mean_uncertainty,
            'Uncertainty_Correlation': uncertainty_correlation
        })
    
    return metrics

def plot_results(y_true, y_pred, uncertainty=None, title="Results", sample_idx=0):
    """결과 시각화"""
    plt.figure(figsize=(15, 8))
    
    # 첫 번째 샘플만 시각화
    true_signal = y_true[sample_idx]
    pred_signal = y_pred[sample_idx]
    
    if len(true_signal.shape) > 1:
        true_signal = true_signal.flatten()
        pred_signal = pred_signal.flatten()
    
    time_axis = np.arange(len(true_signal))
    
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, true_signal, 'b-', label='True Signal', alpha=0.7)
    plt.plot(time_axis, pred_signal, 'r-', label='Predicted Signal', alpha=0.7)
    
    if uncertainty is not None and len(uncertainty) > sample_idx:
        unc = uncertainty[sample_idx]
        if len(unc.shape) > 1:
            unc = unc.flatten()
        plt.fill_between(time_axis, 
                        pred_signal - unc, 
                        pred_signal + unc, 
                        alpha=0.3, label='Uncertainty')
    
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitude')
    plt.title(f'{title} - Signal Comparison')
    plt.legend()
    plt.grid(True)
    
    # 오차 시각화
    plt.subplot(2, 1, 2)
    error = np.abs(true_signal - pred_signal)
    plt.plot(time_axis, error, 'g-', label='Absolute Error')
    
    if uncertainty is not None and len(uncertainty) > sample_idx:
        unc = uncertainty[sample_idx]
        if len(unc.shape) > 1:
            unc = unc.flatten()
        plt.plot(time_axis, unc, 'orange', label='Predicted Uncertainty', alpha=0.7)
    
    plt.xlabel('Time Steps')
    plt.ylabel('Error/Uncertainty')
    plt.title('Error and Uncertainty')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
