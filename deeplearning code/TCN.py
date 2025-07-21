import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Dict

class Chomp1d(nn.Module):
    """Chomp layer to ensure causal convolution"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """Temporal Block with dilated convolutions and residual connections"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
        # Weight normalization
        self.conv1 = nn.utils.weight_norm(self.conv1)
        self.conv2 = nn.utils.weight_norm(self.conv2)
        if self.downsample is not None:
            self.downsample = nn.utils.weight_norm(self.downsample)
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights"""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network"""
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, 
                                   dilation=dilation_size,
                                   padding=(kernel_size-1) * dilation_size, 
                                   dropout=dropout)]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class TCNDenoiser(nn.Module):
    """TCN-based Seismic Signal Denoiser with Uncertainty Quantification"""
    
    def __init__(self, input_size=1, num_channels=[64, 128, 256, 128, 64], 
                 kernel_size=3, dropout=0.2, output_size=1):
        super(TCNDenoiser, self).__init__()
        
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        
        # Additional convolutional layers for feature refinement
        self.feature_conv = nn.Sequential(
            nn.Conv1d(num_channels[-1], num_channels[-1], kernel_size=3, padding=1),
            nn.BatchNorm1d(num_channels[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels[-1], num_channels[-1]//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_channels[-1]//2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layers
        self.prediction_head = nn.Sequential(
            nn.Conv1d(num_channels[-1]//2, num_channels[-1]//4, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels[-1]//4, output_size, kernel_size=1)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Conv1d(num_channels[-1]//2, num_channels[-1]//4, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels[-1]//4, output_size, kernel_size=1),
            nn.Softplus()
        )
        
        # Global context for uncertainty
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_uncertainty = nn.Sequential(
            nn.Linear(num_channels[-1]//2, num_channels[-1]//4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1]//4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, seq_len)
        
        # TCN forward pass
        tcn_out = self.tcn(x)  # (batch_size, num_channels[-1], seq_len)
        
        # Feature refinement
        features = self.feature_conv(tcn_out)  # (batch_size, num_channels[-1]//2, seq_len)
        
        # Predictions
        predictions = self.prediction_head(features).squeeze(1)  # (batch_size, seq_len)
        
        # Local uncertainty
        local_uncertainty = self.uncertainty_head(features).squeeze(1)  # (batch_size, seq_len)
        
        # Global uncertainty context
        global_features = self.global_pool(features).squeeze(-1)  # (batch_size, num_channels[-1]//2)
        global_unc_weight = self.global_uncertainty(global_features)  # (batch_size, 1)
        
        # Combine local and global uncertainty
        uncertainty = local_uncertainty * global_unc_weight
        
        return predictions, uncertainty, features

class MultiScaleTCN(nn.Module):
    """Multi-scale TCN for capturing different temporal patterns"""
    
    def __init__(self, input_size=1, base_channels=64, num_scales=3, 
                 kernel_sizes=[3, 5, 7], dropout=0.2):
        super(MultiScaleTCN, self).__init__()
        
        self.num_scales = num_scales
        
        # Multiple TCN branches with different kernel sizes
        self.tcn_branches = nn.ModuleList()
        
        for i, kernel_size in enumerate(kernel_sizes):
            channels = [base_channels * (2**j) for j in range(4)]  # [64, 128, 256, 512]
            channels += [base_channels * (2**(3-j)) for j in range(1, 4)]  # [256, 128, 64]
            
            tcn = TemporalConvNet(input_size, channels, kernel_size, dropout)
            self.tcn_branches.append(tcn)
        
        # Fusion layer
        total_channels = len(kernel_sizes) * base_channels
        self.fusion = nn.Sequential(
            nn.Conv1d(total_channels, base_channels * 2, kernel_size=1),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(base_channels * 2, base_channels, kernel_size=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism for branch weighting
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(total_channels, total_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(total_channels // 4, len(kernel_sizes), kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # Output layers
        self.prediction_head = nn.Conv1d(base_channels, 1, kernel_size=1)
        self.uncertainty_head = nn.Sequential(
            nn.Conv1d(base_channels, base_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(base_channels // 2, 1, kernel_size=1),
            nn.Softplus()
        )
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        x = x.unsqueeze(1)  # Add channel dimension
        
        # Process through different TCN branches
        branch_outputs = []
        for tcn_branch in self.tcn_branches:
            branch_out = tcn_branch(x)
            branch_outputs.append(branch_out)
        
        # Concatenate branch outputs
        concatenated = torch.cat(branch_outputs, dim=1)
        
        # Compute attention weights
        attn_weights = self.attention(concatenated)  # (batch_size, num_scales, 1)
        
        # Apply attention to each branch
        weighted_branches = []
        start_idx = 0
        for i, branch_out in enumerate(branch_outputs):
            end_idx = start_idx + branch_out.size(1)
            weight = attn_weights[:, i:i+1, :]  # (batch_size, 1, 1)
            weighted_branch = concatenated[:, start_idx:end_idx, :] * weight
            weighted_branches.append(weighted_branch)
            start_idx = end_idx
        
        # Concatenate weighted branches
        weighted_concat = torch.cat(weighted_branches, dim=1)
        
        # Fusion
        fused_features = self.fusion(weighted_concat)
        
        # Generate outputs
        predictions = self.prediction_head(fused_features).squeeze(1)
        uncertainty = self.uncertainty_head(fused_features).squeeze(1)
        
        return predictions, uncertainty, attn_weights.squeeze(-1)

class TCNTrainer:
    """TCN 학습 클래스"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 model_type='standard'):
        self.model = model.to(device)
        self.device = device
        self.model_type = model_type
        self.history = {'train_loss': [], 'val_loss': [], 'train_mse': [], 'val_mse': []}
    
    def uncertainty_loss(self, pred, target, uncertainty, alpha=1.0):
        """Uncertainty-aware loss function"""
        # Heteroscedastic loss
        precision = 1.0 / (uncertainty + 1e-8)
        mse_loss = torch.mean(precision * (pred - target) ** 2)
        uncertainty_reg = torch.mean(torch.log(uncertainty + 1e-8))
        
        total_loss = mse_loss + alpha * uncertainty_reg
        
        return total_loss, mse_loss, uncertainty_reg
    
    def temporal_consistency_loss(self, pred, target, weight=0.1):
        """Temporal consistency regularization"""
        # First-order temporal difference
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        
        consistency_loss = F.mse_loss(pred_diff, target_diff)
        
        return weight * consistency_loss
    
    def train_epoch(self, train_loader, optimizer, alpha=1.0, beta=0.1):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0
        total_mse = 0
        total_unc_loss = 0
        total_temp_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            
            optimizer.zero_grad()
            
            if self.model_type == 'multiscale':
                pred, uncertainty, _ = self.model(data)
            else:
                pred, uncertainty, _ = self.model(data)
            
            # Main loss
            loss, mse_loss, unc_loss = self.uncertainty_loss(pred, target, uncertainty, alpha)
            
            # Temporal consistency loss
            temp_loss = self.temporal_consistency_loss(pred, target, beta)
            
            total_loss_batch = loss + temp_loss
            
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            total_mse += mse_loss.item()
            total_unc_loss += unc_loss.item()
            total_temp_loss += temp_loss.item()
        
        return (total_loss / len(train_loader), 
                total_mse / len(train_loader), 
                total_unc_loss / len(train_loader),
                total_temp_loss / len(train_loader))
    
    def validate(self, val_loader, alpha=1.0, beta=0.1):
        """검증"""
        self.model.eval()
        total_loss = 0
        total_mse = 0
        total_unc_loss = 0
        total_temp_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                
                if self.model_type == 'multiscale':
                    pred, uncertainty, _ = self.model(data)
                else:
                    pred, uncertainty, _ = self.model(data)
                
                loss, mse_loss, unc_loss = self.uncertainty_loss(pred, target, uncertainty, alpha)
                temp_loss = self.temporal_consistency_loss(pred, target, beta)
                
                total_loss_batch = loss + temp_loss
                
                total_loss += total_loss_batch.item()
                total_mse += mse_loss.item()
                total_unc_loss += unc_loss.item()
                total_temp_loss += temp_loss.item()
        
        return (total_loss / len(val_loader), 
                total_mse / len(val_loader), 
                total_unc_loss / len(val_loader),
                total_temp_loss / len(val_loader))
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, alpha=1.0, beta=0.1,
              patience=15, save_path='best_tcn.pth'):
        """전체 학습 프로세스"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.7)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting TCN ({self.model_type}) training...")
        
        for epoch in range(epochs):
            # Training
            train_loss, train_mse, train_unc, train_temp = self.train_epoch(
                train_loader, optimizer, alpha, beta
            )
            
            # Validation
            val_loss, val_mse, val_unc, val_temp = self.validate(val_loader, alpha, beta)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mse'].append(train_mse)
            self.history['val_mse'].append(val_mse)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.6f} (MSE: {train_mse:.6f}, Unc: {train_unc:.6f}, Temp: {train_temp:.6f})')
            print(f'  Val Loss: {val_loss:.6f} (MSE: {val_mse:.6f}, Unc: {val_unc:.6f}, Temp: {val_temp:.6f})')
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
        attention_weights = []
        
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
                    if self.model_type == 'multiscale':
                        pred, unc, attn = self.model(data)
                        mc_attentions.append(attn.cpu().numpy())
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
                if mc_attentions:
                    attention_weights.append(np.mean(mc_attentions, axis=0))
        
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
            'attention_weights': attention_weights if attention_weights else None
        }

def run_tcn_experiment(data_dict, model_type='standard', batch_size=32, 
                      epochs=100, learning_rate=0.001):
    """TCN 실험 실행 함수"""
    
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
    
    if model_type == 'multiscale':
        model = MultiScaleTCN(
            input_size=1,
            base_channels=64,
            num_scales=3,
            kernel_sizes=[3, 5, 7],
            dropout=0.2
        )
    else:
        model = TCNDenoiser(
            input_size=1,
            num_channels=[64, 128, 256, 128, 64],
            kernel_size=3,
            dropout=0.2
        )
    
    trainer = TCNTrainer(model, device, model_type=model_type)
    
    # 학습
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=learning_rate,
        alpha=1.0,  # Uncertainty loss weight
        beta=0.1,   # Temporal consistency weight
        patience=15,
        save_path=f'best_tcn_{model_type}.pth'
    )
    
    # 최고 모델 로드
    checkpoint = torch.load(f'best_tcn_{model_type}.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 테스트 및 불확실성 추정
    print("\nRunning inference with uncertainty quantification...")
    results = trainer.predict_with_uncertainty(test_loader, n_samples=50)
    
    # 메트릭 계산
    predictions = results['predictions']
    targets = results['targets']
    total_uncertainty = results['total_uncertainty']
    
    metrics = calculate_metrics(targets, predictions, total_uncertainty)
    
    print(f"\n=== TCN ({model_type}) Results ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")
    
    # 결과 시각화
    plot_results(
        targets, predictions, total_uncertainty, 
        title=f"TCN ({model_type}) Results", sample_idx=0
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

# Standard TCN 실험
tcn_results = run_tcn_experiment(
    data_dict=data_dict,
    model_type='standard',
    batch_size=32,
    epochs=100,
    learning_rate=0.001
)

# Multi-scale TCN 실험
multiscale_tcn_results = run_tcn_experiment(
    data_dict=data_dict,
    model_type='multiscale',
    batch_size=32,
    epochs=100,
    learning_rate=0.001
)
"""
