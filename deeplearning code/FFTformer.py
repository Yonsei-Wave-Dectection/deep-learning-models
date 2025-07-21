import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import math
from typing import Tuple, Dict

class FourierTransform(nn.Module):
    """Learnable Fourier Transform Layer"""
    def __init__(self, seq_len, modes=32):
        super(FourierTransform, self).__init__()
        self.seq_len = seq_len
        self.modes = min(modes, seq_len // 2)
        
        # Learnable Fourier weights
        self.weights_real = nn.Parameter(torch.randn(self.modes, dtype=torch.float32) * 0.02)
        self.weights_imag = nn.Parameter(torch.randn(self.modes, dtype=torch.float32) * 0.02)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        
        # Apply FFT
        x_ft = torch.fft.rfft(x, dim=1, norm='ortho')
        
        # Apply learnable weights only to lower modes
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :self.modes] = x_ft[:, :self.modes] * torch.complex(self.weights_real[:self.modes], 
                                                                       self.weights_imag[:self.modes]).unsqueeze(0).unsqueeze(-1)
        
        # Inverse FFT
        x = torch.fft.irfft(out_ft, n=seq_len, dim=1, norm='ortho')
        
        return x

class FourierBlock(nn.Module):
    """Fourier Block combining FFT and MLP"""
    def __init__(self, d_model, seq_len, modes=32, dropout=0.1):
        super(FourierBlock, self).__init__()
        
        self.fourier_layer = FourierTransform(seq_len, modes)
        
        # MLP for non-linear transformation
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Fourier transformation with residual connection
        x_fourier = self.fourier_layer(self.norm1(x))
        x = x + self.dropout(x_fourier)
        
        # MLP with residual connection
        x_mlp = self.mlp(self.norm2(x))
        x = x + self.dropout(x_mlp)
        
        return x

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, d_model = query.shape
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        output = self.w_o(context)
        
        return output, attention_weights

class FFTformerBlock(nn.Module):
    """FFTformer Block combining Fourier Transform and Self-Attention"""
    def __init__(self, d_model, seq_len, n_heads=8, modes=32, dropout=0.1):
        super(FFTformerBlock, self).__init__()
        
        # Fourier branch
        self.fourier_block = FourierBlock(d_model, seq_len, modes, dropout)
        
        # Attention branch
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Fusion and normalization
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Feed forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm_ffn = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Store input for residual
        residual = x
        
        # Fourier branch
        x_fourier = self.fourier_block(x)
        
        # Attention branch
        x_attention, attention_weights = self.self_attention(x, x, x)
        
        # Combine branches
        x_combined = torch.cat([x_fourier, x_attention], dim=-1)
        x_fused = self.fusion(x_combined)
        
        # Residual connection and normalization
        x = self.norm(residual + self.dropout(x_fused))
        
        # Feed forward with residual
        x_ffn = self.ffn(self.norm_ffn(x))
        x = x + self.dropout(x_ffn)
        
        return x, attention_weights

class FFTformer(nn.Module):
    """FFTformer for Seismic Signal Denoising with Uncertainty Quantification"""
    
    def __init__(self, seq_len=1000, d_model=256, n_layers=6, n_heads=8, 
                 modes=64, dropout=0.1):
        super(FFTformer, self).__init__()
        
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, seq_len)
        
        # FFTformer layers
        self.layers = nn.ModuleList([
            FFTformerBlock(d_model, seq_len, n_heads, modes, dropout)
            for _ in range(n_layers)
        ])
        
        # Output layers
        self.norm_final = nn.LayerNorm(d_model)
        
        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
        # Uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
            nn.Softplus()
        )
        
        # Frequency-domain uncertainty
        self.freq_uncertainty = nn.Parameter(torch.ones(modes) * 0.1)
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        batch_size, seq_len = x.shape
        
        # Project to model dimension
        x = x.unsqueeze(-1)  # (batch_size, seq_len, 1)
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Store attention weights
        attention_weights = []
        
        # Pass through FFTformer layers
        for layer in self.layers:
            x, attn_weights = layer(x)
            attention_weights.append(attn_weights)
        
        # Final normalization
        x = self.norm_final(x)
        
        # Generate predictions and uncertainty
        predictions = self.prediction_head(x).squeeze(-1)  # (batch_size, seq_len)
        uncertainty = self.uncertainty_head(x).squeeze(-1)  # (batch_size, seq_len)
        
        return predictions, uncertainty, attention_weights

class SpectralFFTformer(nn.Module):
    """Spectral FFTformer with frequency domain processing"""
    
    def __init__(self, seq_len=1000, d_model=256, n_layers=4, modes=64, dropout=0.1):
        super(SpectralFFTformer, self).__init__()
        
        self.seq_len = seq_len
        self.d_model = d_model
        self.modes = min(modes, seq_len // 2)
        
        # Spectral embedding
        self.spectral_embedding = nn.Linear(2, d_model)  # Real and imaginary parts
        
        # Spectral transformer layers
        self.spectral_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            for _ in range(n_layers)
        ])
        
        # Spectral output
        self.spectral_output = nn.Linear(d_model, 2)  # Real and imaginary
        
        # Uncertainty in frequency domain
        self.freq_uncertainty = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus()
        )
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        batch_size, seq_len = x.shape
        
        # Convert to frequency domain
        x_freq = torch.fft.rfft(x, dim=1, norm='ortho')  # (batch_size, freq_bins)
        
        # Take only low frequency modes
        x_freq = x_freq[:, :self.modes]
        
        # Convert complex to real representation
        x_real = torch.stack([x_freq.real, x_freq.imag], dim=-1)  # (batch_size, modes, 2)
        
        # Embed spectral features
        x_embedded = self.spectral_embedding(x_real)  # (batch_size, modes, d_model)
        
        # Process through transformer layers
        for layer in self.spectral_layers:
            x_embedded = layer(x_embedded)
        
        # Generate spectral output
        spectral_out = self.spectral_output(x_embedded)  # (batch_size, modes, 2)
        freq_uncertainty = self.freq_uncertainty(x_embedded).squeeze(-1)  # (batch_size, modes)
        
        # Convert back to complex
        complex_out = torch.complex(spectral_out[..., 0], spectral_out[..., 1])
        
        # Pad and convert back to time domain
        full_freq = torch.zeros(batch_size, seq_len // 2 + 1, dtype=torch.complex64, device=x.device)
        full_freq[:, :self.modes] = complex_out
        
        # Inverse FFT
        time_domain = torch.fft.irfft(full_freq, n=seq_len, dim=1, norm='ortho')
        
        # Uncertainty in time domain (using learnable transformation)
        time_uncertainty = torch.zeros(batch_size, seq_len, device=x.device)
        time_uncertainty[:, :self.modes] = freq_uncertainty
        
        return time_domain, time_uncertainty

class FFTformerTrainer:
    """FFTformer 학습 클래스"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 model_type='standard'):
        self.model = model.to(device)
        self.device = device
        self.model_type = model_type
        self.history = {'train_loss': [], 'val_loss': [], 'train_mse': [], 'val_mse': []}
    
    def spectral_loss(self, pred, target, pred_uncertainty=None, alpha=1.0):
        """Spectral domain loss function"""
        # Time domain loss
        time_loss = F.mse_loss(pred, target)
        
        # Frequency domain loss
        pred_freq = torch.fft.rfft(pred, dim=1, norm='ortho')
        target_freq = torch.fft.rfft(target, dim=1, norm='ortho')
        
        freq_loss = F.mse_loss(pred_freq.real, target_freq.real) + \
                   F.mse_loss(pred_freq.imag, target_freq.imag)
        
        total_loss = time_loss + 0.1 * freq_loss
        
        # Add uncertainty regularization
        if pred_uncertainty is not None:
            precision = 1.0 / (pred_uncertainty + 1e-8)
            uncertainty_loss = torch.mean(precision * (pred - target) ** 2)
            uncertainty_reg = torch.mean(torch.log(pred_uncertainty + 1e-8))
            total_loss = uncertainty_loss + alpha * uncertainty_reg
        
        return total_loss, time_loss, freq_loss
    
    def train_epoch(self, train_loader, optimizer, alpha=1.0):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0
        total_time_loss = 0
        total_freq_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            
            optimizer.zero_grad()
            
            if self.model_type == 'spectral':
                pred, uncertainty = self.model(data)
                loss, time_loss, freq_loss = self.spectral_loss(pred, target, uncertainty, alpha)
            else:
                pred, uncertainty, _ = self.model(data)
                loss, time_loss, freq_loss = self.spectral_loss(pred, target, uncertainty, alpha)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_time_loss += time_loss.item()
            total_freq_loss += freq_loss.item()
        
        return (total_loss / len(train_loader), 
                total_time_loss / len(train_loader), 
                total_freq_loss / len(train_loader))
    
    def validate(self, val_loader, alpha=1.0):
        """검증"""
        self.model.eval()
        total_loss = 0
        total_time_loss = 0
        total_freq_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                
                if self.model_type == 'spectral':
                    pred, uncertainty = self.model(data)
                    loss, time_loss, freq_loss = self.spectral_loss(pred, target, uncertainty, alpha)
                else:
                    pred, uncertainty, _ = self.model(data)
                    loss, time_loss, freq_loss = self.spectral_loss(pred, target, uncertainty, alpha)
                
                total_loss += loss.item()
                total_time_loss += time_loss.item()
                total_freq_loss += freq_loss.item()
        
        return (total_loss / len(val_loader), 
                total_time_loss / len(val_loader), 
                total_freq_loss / len(val_loader))
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.0001, alpha=1.0,
              patience=15, save_path='best_fftformer.pth'):
        """전체 학습 프로세스"""
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.1)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting FFTformer ({self.model_type}) training...")
        
        for epoch in range(epochs):
            # Training
            train_loss, train_time, train_freq = self.train_epoch(train_loader, optimizer, alpha)
            
            # Validation
            val_loss, val_time, val_freq = self.validate(val_loader, alpha)
            
            # Learning rate scheduling
            scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mse'].append(train_time)
            self.history['val_mse'].append(val_time)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.6f} (Time: {train_time:.6f}, Freq: {train_freq:.6f})')
            print(f'  Val Loss: {val_loss:.6f} (Time: {val_time:.6f}, Freq: {val_freq:.6f})')
            print(f'  LR: {optimizer.param_groups[0]["lr"]:.8f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'epoch': epoch,
                    'history': self.history
                }, save_path)
                print(f'  New best model saved! Val Loss: {val_loss:.6f}')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
            
            print('-' * 60)
        
        print("Training completed!")
        return self.history
    
    def predict_with_uncertainty(self, test_loader, n_samples=50):
        """불확실성을 포함한 예측"""
        predictions = []
        uncertainties = []
        epistemic_uncertainties = []
        targets = []
        attention_weights_all = []
        
        # Monte Carlo Dropout for epistemic uncertainty
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Multiple forward passes
                mc_predictions = []
                mc_uncertainties = []
                mc_attentions = []
                
                for _ in range(n_samples):
                    if self.model_type == 'spectral':
                        pred, unc = self.model(data)
                        attn = None
                    else:
                        pred, unc, attn = self.model(data)
                    
                    mc_predictions.append(pred.cpu().numpy())
                    mc_uncertainties.append(unc.cpu().numpy())
                    if attn is not None:
                        mc_attentions.append([a.cpu().numpy() for a in attn])
                
                mc_predictions = np.array(mc_predictions)
                mc_uncertainties = np.array(mc_uncertainties)
                
                # Statistics
                mean_pred = np.mean(mc_predictions, axis=0)
                epistemic_unc = np.std(mc_predictions, axis=0)
                aleatoric_unc = np.mean(mc_uncertainties, axis=0)
                
                predictions.append(mean_pred)
                uncertainties.append(aleatoric_unc)
                epistemic_uncertainties.append(epistemic_unc)
                targets.append(target.cpu().numpy())
                if mc_attentions:
                    attention_weights_all.append(mc_attentions)
        
        # Concatenate results
        predictions = np.concatenate(predictions, axis=0)
        aleatoric_uncertainty = np.concatenate(uncertainties, axis=0)
        epistemic_uncertainty = np.concatenate(epistemic_uncertainties, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        # Total uncertainty
        total_uncertainty = np.sqrt(aleatoric_uncertainty**2 + epistemic_uncertainty**2)
        
        return {
            'predictions': predictions,
            'targets': targets,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'epistemic_uncertainty': epistemic_uncertainty,
            'total_uncertainty': total_uncertainty,
            'attention_weights': attention_weights_all if attention_weights_all else None
        }

def run_fftformer_experiment(data_dict, model_type='standard', batch_size=32, 
                            epochs=100, learning_rate=0.0001):
    """FFTformer 실험 실행 함수"""
    
    # 데이터셋 및 데이터로더 생성
    train_dataset = SeismicDataset(data_dict['X_train'], data_dict['y_train'])
    val_dataset = SeismicDataset(data_dict['X_val'], data_dict['y_val'])
    test_dataset = SeismicDataset(data_dict['X_test'], data_dict['y_test'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 모델 생성
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    seq_len = data_dict['X_train'].shape[1]
    
    if model_type == 'spectral':
        model = SpectralFFTformer(
            seq_len=seq_len,
            d_model=256,
            n_layers=4,
            modes=64,
            dropout=0.1
        )
    else:
        model = FFTformer(
            seq_len=seq_len,
            d_model=256,
            n_layers=6,
            n_heads=8,
            modes=64,
            dropout=0.1
        )
    
    trainer = FFTformerTrainer(model, device, model_type=model_type)
    
    # 학습
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=learning_rate,
        alpha=1.0,
        patience=15,
        save_path=f'best_fftformer_{model_type}.pth'
    )
    
    # 최고 모델 로드
    checkpoint = torch.load(f'best_fftformer_{model_type}.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 테스트 및 불확실성 추정
    print("\nRunning inference with uncertainty quantification...")
    results = trainer.predict_with_uncertainty(test_loader, n_samples=50)
    
    # 메트릭 계산
    predictions = results['predictions']
    targets = results['targets']
    total_uncertainty = results['total_uncertainty']
    
    metrics = calculate_metrics(targets, predictions, total_uncertainty)
    
    print(f"\n=== FFTformer ({model_type}) Results ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")
    
    # 결과 시각화
    plot_results(
        targets, predictions, total_uncertainty, 
        title=f"FFTformer ({model_type}) Results", sample_idx=0
    )
    
    return {
        'model': model,
        'trainer': trainer,
        'history': history,
        'results': results,
        'metrics': metrics
    }

# 사용 예시:
"""
# 데이터 준비
data_loader = SeismicDataLoader(csv_dir="/path/to/preprocessed_csv/", sequence_length=1000)
data_dict = data_loader.prepare_data(test_size=0.2, validation_size=0.1, noise_factor=0.1)

# Standard FFTformer 실험
fftformer_results = run_fftformer_experiment(
    data_dict=data_dict,
    model_type='standard',
    batch_size=32,
    epochs=100,
    learning_rate=0.0001
)

# Spectral FFTformer 실험
spectral_fftformer_results = run_fftformer_experiment(
    data_dict=data_dict,
    model_type='spectral',
    batch_size=32,
    epochs=100,
    learning_rate=0.0001
)
"""
