import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import math
from typing import Tuple, Dict

class CausalConv1d(nn.Module):
    """Causal 1D Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             padding=self.padding, dilation=dilation)
    
    def forward(self, x):
        x = self.conv(x)
        if self.padding != 0:
            x = x[:, :, :-self.padding]
        return x

class ResidualBlock(nn.Module):
    """WaveNet Residual Block with Gated Activation"""
    def __init__(self, residual_channels, gate_channels, skip_channels, 
                 kernel_size, dilation, dropout=0.2):
        super(ResidualBlock, self).__init__()
        
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        
        # Dilated causal convolution
        self.dilated_conv = CausalConv1d(residual_channels, gate_channels, 
                                        kernel_size, dilation)
        
        # Gated activation unit
        self.filter_conv = nn.Conv1d(gate_channels, residual_channels, 1)
        self.gate_conv = nn.Conv1d(gate_channels, residual_channels, 1)
        
        # Skip connection
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, 1)
        
        # Residual connection
        self.residual_conv = nn.Conv1d(residual_channels, residual_channels, 1)
        
        # Batch normalization and dropout
        self.bn = nn.BatchNorm1d(residual_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Dilated convolution
        dilated_out = self.dilated_conv(x)
        
        # Gated activation: tanh(filter) * sigmoid(gate)
        filter_out = torch.tanh(self.filter_conv(dilated_out))
        gate_out = torch.sigmoid(self.gate_conv(dilated_out))
        gated = filter_out * gate_out
        
        # Apply dropout and batch norm
        gated = self.dropout(self.bn(gated))
        
        # Skip connection
        skip = self.skip_conv(gated)
        
        # Residual connection
        residual = self.residual_conv(gated)
        residual = residual + x  # Add input for residual connection
        
        return residual, skip

class WaveNet(nn.Module):
    """WaveNet for Seismic Signal Denoising with Uncertainty Quantification"""
    
    def __init__(self, layers=10, blocks=4, dilation_channels=32, residual_channels=32,
                 skip_channels=256, end_channels=256, kernel_size=2, dropout=0.2):
        super(WaveNet, self).__init__()
        
        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        
        # Input layer
        self.start_conv = nn.Conv1d(1, residual_channels, 1)
        
        # Receptive field calculation
        self.receptive_field = 1
        for b in range(blocks):
            for l in range(layers):
                self.receptive_field += (kernel_size - 1) * (2 ** l)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for b in range(blocks):
            for l in range(layers):
                dilation = 2 ** l
                self.residual_blocks.append(
                    ResidualBlock(residual_channels, dilation_channels, 
                                skip_channels, kernel_size, dilation, dropout)
                )
        
        # Post-processing layers
        self.end_conv_1 = nn.Conv1d(skip_channels, end_channels, 1)
        self.end_conv_2 = nn.Conv1d(end_channels, end_channels, 1)
        
        # Output layers
        self.prediction_head = nn.Sequential(
            nn.Conv1d(end_channels, end_channels // 2, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(end_channels // 2, end_channels // 4, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(end_channels // 4, 1, 1)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Conv1d(end_channels, end_channels // 2, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(end_channels // 2, end_channels // 4, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(end_channels // 4, 1, 1),
            nn.Softplus()
        )
        
        # Global uncertainty context
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(end_channels, end_channels // 4, 1),
            nn.ReLU(),
            nn.Conv1d(end_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (batch_size, seq_len)
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, seq_len)
        
        # Initial convolution
        x = self.start_conv(x)
        
        # Collect skip connections
        skip_connections = []
        
        # Pass through residual blocks
        for block in self.residual_blocks:
            x, skip = block(x)
            skip_connections.append(skip)
        
        # Sum all skip connections
        skip_sum = torch.stack(skip_connections, dim=0).sum(dim=0)
        
        # Post-processing
        x = F.relu(skip_sum)
        x = F.relu(self.end_conv_1(x))
        x = F.relu(self.end_conv_2(x))
        
        # Generate predictions
        predictions = self.prediction_head(x).squeeze(1)
        
        # Generate uncertainty
        local_uncertainty = self.uncertainty_head(x).squeeze(1)
        global_context = self.global_context(x).squeeze(1)
        
        # Combine local and global uncertainty
        uncertainty = local_uncertainty * global_context
        
        return predictions, uncertainty, skip_connections

class ConditionalWaveNet(nn.Module):
    """Conditional WaveNet with local conditioning"""
    
    def __init__(self, layers=10, blocks=4, dilation_channels=32, residual_channels=32,
                 skip_channels=256, end_channels=256, condition_channels=128, 
                 kernel_size=2, dropout=0.2):
        super(ConditionalWaveNet, self).__init__()
        
        self.condition_channels = condition_channels
        
        # Conditioning network for extracting local features
        self.condition_net = nn.Sequential(
            nn.Conv1d(1, condition_channels // 4, kernel_size=7, padding=3),
            nn.BatchNorm1d(condition_channels // 4),
            nn.ReLU(),
            nn.Conv1d(condition_channels // 4, condition_channels // 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(condition_channels // 2),
            nn.ReLU(),
            nn.Conv1d(condition_channels // 2, condition_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(condition_channels),
            nn.ReLU()
        )
        
        # Input layer
        self.start_conv = nn.Conv1d(1, residual_channels, 1)
        
        # Conditional residual blocks
        self.residual_blocks = nn.ModuleList()
        for b in range(blocks):
            for l in range(layers):
                dilation = 2 ** l
                self.residual_blocks.append(
                    ConditionalResidualBlock(residual_channels, dilation_channels, 
                                           skip_channels, condition_channels,
                                           kernel_size, dilation, dropout)
                )
        
        # Post-processing
        self.end_conv_1 = nn.Conv1d(skip_channels, end_channels, 1)
        self.end_conv_2 = nn.Conv1d(end_channels, end_channels, 1)
        
        # Output layers
        self.prediction_head = nn.Conv1d(end_channels, 1, 1)
        self.uncertainty_head = nn.Sequential(
            nn.Conv1d(end_channels, end_channels // 2, 1),
            nn.ReLU(),
            nn.Conv1d(end_channels // 2, 1, 1),
            nn.Softplus()
        )
    
    def forward(self, x):
        # Extract conditioning information
        condition = self.condition_net(x.unsqueeze(1))
        
        # Initial convolution
        x = x.unsqueeze(1)
        x = self.start_conv(x)
        
        # Collect skip connections
        skip_connections = []
        
        # Pass through conditional residual blocks
        for block in self.residual_blocks:
            x, skip = block(x, condition)
            skip_connections.append(skip)
        
        # Sum skip connections
        skip_sum = torch.stack(skip_connections, dim=0).sum(dim=0)
        
        # Post-processing
        x = F.relu(skip_sum)
        x = F.relu(self.end_conv_1(x))
        x = F.relu(self.end_conv_2(x))
        
        # Generate outputs
        predictions = self.prediction_head(x).squeeze(1)
        uncertainty = self.uncertainty_head(x).squeeze(1)
        
        return predictions, uncertainty

class ConditionalResidualBlock(nn.Module):
    """Conditional Residual Block with local conditioning"""
    
    def __init__(self, residual_channels, gate_channels, skip_channels, 
                 condition_channels, kernel_size, dilation, dropout=0.2):
        super(ConditionalResidualBlock, self).__init__()
        
        # Main dilated convolution
        self.dilated_conv = CausalConv1d(residual_channels, gate_channels, 
                                        kernel_size, dilation)
        
        # Conditioning convolutions
        self.condition_conv = nn.Conv1d(condition_channels, gate_channels, 1)
        
        # Gated activation
        self.filter_conv = nn.Conv1d(gate_channels, residual_channels, 1)
        self.gate_conv = nn.Conv1d(gate_channels, residual_channels, 1)
        
        # Output convolutions
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, 1)
        self.residual_conv = nn.Conv1d(residual_channels, residual_channels, 1)
        
        # Normalization and dropout
        self.bn = nn.BatchNorm1d(residual_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, condition):
        # Dilated convolution
        dilated_out = self.dilated_conv(x)
        
        # Add conditioning
        condition_out = self.condition_conv(condition)
        combined = dilated_out + condition_out
        
        # Gated activation
        filter_out = torch.tanh(self.filter_conv(combined))
        gate_out = torch.sigmoid(self.gate_conv(combined))
        gated = filter_out * gate_out
        
        # Normalization and dropout
        gated = self.dropout(self.bn(gated))
        
        # Output connections
        skip = self.skip_conv(gated)
        residual = self.residual_conv(gated) + x
        
        return residual, skip

class WaveNetTrainer:
    """WaveNet 학습 클래스"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 model_type='standard'):
        self.model = model.to(device)
        self.device = device
        self.model_type = model_type
        self.history = {'train_loss': [], 'val_loss': [], 'train_mse': [], 'val_mse': []}
    
    def wavenet_loss(self, pred, target, uncertainty, alpha=1.0, beta=0.1):
        """WaveNet-specific loss with uncertainty and waveform preservation"""
        # Main reconstruction loss
        mse_loss = F.mse_loss(pred, target)
        
        # Uncertainty-aware loss
        precision = 1.0 / (uncertainty + 1e-8)
        uncertainty_loss = torch.mean(precision * (pred - target) ** 2)
        uncertainty_reg = torch.mean(torch.log(uncertainty + 1e-8))
        
        # Waveform coherence loss (gradient matching)
        pred_grad = pred[:, 1:] - pred[:, :-1]
        target_grad = target[:, 1:] - target[:, :-1]
        gradient_loss = F.mse_loss(pred_grad, target_grad)
        
        # Spectral loss
        pred_fft = torch.fft.rfft(pred, dim=1)
        target_fft = torch.fft.rfft(target, dim=1)
        spectral_loss = F.mse_loss(pred_fft.real, target_fft.real) + \
                       F.mse_loss(pred_fft.imag, target_fft.imag)
        
        total_loss = uncertainty_loss + alpha * uncertainty_reg + \
                    beta * gradient_loss + 0.1 * spectral_loss
        
        return total_loss, mse_loss, uncertainty_reg, gradient_loss, spectral_loss
    
    def train_epoch(self, train_loader, optimizer, alpha=1.0, beta=0.1):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0
        total_mse = 0
        total_unc_reg = 0
        total_grad_loss = 0
        total_spec_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            
            optimizer.zero_grad()
            
            if self.model_type == 'conditional':
                pred, uncertainty = self.model(data)
            else:
                pred, uncertainty, _ = self.model(data)
            
            loss, mse_loss, unc_reg, grad_loss, spec_loss = self.wavenet_loss(
                pred, target, uncertainty, alpha, beta
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_unc_reg += unc_reg.item()
            total_grad_loss += grad_loss.item()
            total_spec_loss += spec_loss.item()
        
        return (total_loss / len(train_loader), 
                total_mse / len(train_loader),
                total_unc_reg / len(train_loader),
                total_grad_loss / len(train_loader),
                total_spec_loss / len(train_loader))
    
    def validate(self, val_loader, alpha=1.0, beta=0.1):
        """검증"""
        self.model.eval()
        total_loss = 0
        total_mse = 0
        total_unc_reg = 0
        total_grad_loss = 0
        total_spec_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                
                if self.model_type == 'conditional':
                    pred, uncertainty = self.model(data)
                else:
                    pred, uncertainty, _ = self.model(data)
                
                loss, mse_loss, unc_reg, grad_loss, spec_loss = self.wavenet_loss(
                    pred, target, uncertainty, alpha, beta
                )
                
                total_loss += loss.item()
                total_mse += mse_loss.item()
                total_unc_reg += unc_reg.item()
                total_grad_loss += grad_loss.item()
                total_spec_loss += spec_loss.item()
        
        return (total_loss / len(val_loader), 
                total_mse / len(val_loader),
                total_unc_reg / len(val_loader),
                total_grad_loss / len(val_loader),
                total_spec_loss / len(val_loader))
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, alpha=1.0, beta=0.1,
              patience=15, save_path='best_wavenet.pth'):
        """전체 학습 프로세스"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.7)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting WaveNet ({self.model_type}) training...")
        
        for epoch in range(epochs):
            # Training
            train_loss, train_mse, train_unc, train_grad, train_spec = self.train_epoch(
                train_loader, optimizer, alpha, beta
            )
            
            # Validation
            val_loss, val_mse, val_unc, val_grad, val_spec = self.validate(
                val_loader, alpha, beta
            )
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mse'].append(train_mse)
            self.history['val_mse'].append(val_mse)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.6f} (MSE: {train_mse:.6f}, Unc: {train_unc:.6f}, Grad: {train_grad:.6f}, Spec: {train_spec:.6f})')
            print(f'  Val Loss: {val_loss:.6f} (MSE: {val_mse:.6f}, Unc: {val_unc:.6f}, Grad: {val_grad:.6f}, Spec: {val_spec:.6f})')
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
        
        # Monte Carlo Dropout
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
                
                for _ in range(n_samples):
                    if self.model_type == 'conditional':
                        pred, unc = self.model(data)
                    else:
                        pred, unc, _ = self.model(data)
                    
                    mc_predictions.append(pred.cpu().numpy())
                    mc_uncertainties.append(unc.cpu().numpy())
                
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
            'total_uncertainty': total_uncertainty
        }

def run_wavenet_experiment(data_dict, model_type='standard', batch_size=32, 
                          epochs=100, learning_rate=0.001):
    """WaveNet 실험 실행 함수"""
    
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
    
    if model_type == 'conditional':
        model = ConditionalWaveNet(
            layers=10,
            blocks=4,
            dilation_channels=32,
            residual_channels=32,
            skip_channels=256,
            end_channels=256,
            condition_channels=128,
            kernel_size=2,
            dropout=0.2
        )
    else:
        model = WaveNet(
            layers=10,
            blocks=4,
            dilation_channels=32,
            residual_channels=32,
            skip_channels=256,
            end_channels=256,
            kernel_size=2,
            dropout=0.2
        )
    
    trainer = WaveNetTrainer(model, device, model_type=model_type)
    
    # 학습
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=learning_rate,
        alpha=1.0,  # Uncertainty regularization weight
        beta=0.1,   # Gradient loss weight
        patience=15,
        save_path=f'best_wavenet_{model_type}.pth'
    )
    
    # 최고 모델 로드
    checkpoint = torch.load(f'best_wavenet_{model_type}.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 테스트 및 불확실성 추정
    print("\nRunning inference with uncertainty quantification...")
    results = trainer.predict_with_uncertainty(test_loader, n_samples=50)
    
    # 메트릭 계산
    predictions = results['predictions']
    targets = results['targets']
    total_uncertainty = results['total_uncertainty']
    
    metrics = calculate_metrics(targets, predictions, total_uncertainty)
    
    print(f"\n=== WaveNet ({model_type}) Results ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")
    
    # 결과 시각화
    plot_results(
        targets, predictions, total_uncertainty, 
        title=f"WaveNet ({model_type}) Results", sample_idx=0
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

# Standard WaveNet 실험
wavenet_results = run_wavenet_experiment(
    data_dict=data_dict,
    model_type='standard',
    batch_size=32,
    epochs=100,
    learning_rate=0.001
)

# Conditional WaveNet 실험
conditional_wavenet_results = run_wavenet_experiment(
    data_dict=data_dict,
    model_type='conditional',
    batch_size=32,
    epochs=100,
    learning_rate=0.001
)
"""
