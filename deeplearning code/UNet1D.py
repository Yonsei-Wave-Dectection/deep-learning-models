import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple

class DoubleConv1D(nn.Module):
    """1D Double Convolution Block"""
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super(DoubleConv1D, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout_p),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout_p)
        )
    
    def forward(self, x):
        return self.conv_block(x)

class UNet1D(nn.Module):
    """1D U-Net for Seismic Signal Denoising with Uncertainty Quantification"""
    
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], dropout_p=0.1):
        super(UNet1D, self).__init__()
        self.features = features
        self.dropout_p = dropout_p
        
        # Encoder (Contracting Path)
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        for feature in features:
            self.encoder.append(DoubleConv1D(in_channels, feature, dropout_p))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = DoubleConv1D(features[-1], features[-1] * 2, dropout_p)
        
        # Decoder (Expanding Path)
        self.decoder = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        
        for feature in reversed(features):
            self.upconvs.append(nn.ConvTranspose1d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv1D(feature * 2, feature, dropout_p))
        
        # Final output layer
        self.final_conv = nn.Conv1d(features[0], out_channels, kernel_size=1)
        
        # Uncertainty estimation layers
        self.uncertainty_conv = nn.Sequential(
            nn.Conv1d(features[0], features[0] // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout_p),
            nn.Conv1d(features[0] // 2, out_channels, kernel_size=1),
            nn.Softplus()  # Ensure positive uncertainty values
        )
    
    def forward(self, x):
        # Store skip connections
        skip_connections = []
        
        # Encoder
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Reverse skip connections for decoder
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for idx, (upconv, decoder_layer) in enumerate(zip(self.upconvs, self.decoder)):
            x = upconv(x)
            skip_connection = skip_connections[idx]
            
            # Handle size mismatch due to pooling
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2])
            
            # Concatenate skip connection
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = decoder_layer(concat_skip)
        
        # Get features before final layer for uncertainty estimation
        features = x
        
        # Final prediction
        prediction = self.final_conv(x)
        
        # Uncertainty estimation
        uncertainty = self.uncertainty_conv(features)
        
        return prediction, uncertainty

class UNet1DTrainer:
    """1D U-Net 학습 클래스"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {'train_loss': [], 'val_loss': []}
    
    def uncertainty_loss(self, pred, target, uncertainty, alpha=1.0):
        """불확실성을 고려한 손실 함수"""
        # Heteroscedastic loss (aleatoric uncertainty)
        precision = 1.0 / (uncertainty + 1e-8)
        mse_loss = torch.mean(precision * (pred - target) ** 2)
        uncertainty_loss = torch.mean(torch.log(uncertainty + 1e-8))
        
        total_loss = mse_loss + alpha * uncertainty_loss
        return total_loss, mse_loss, uncertainty_loss
    
    def train_epoch(self, train_loader, optimizer, alpha=1.0):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0
        total_mse = 0
        total_unc_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.unsqueeze(1).to(self.device)  # Add channel dimension
            target = target.unsqueeze(1).to(self.device)
            
            optimizer.zero_grad()
            
            pred, uncertainty = self.model(data)
            loss, mse_loss, unc_loss = self.uncertainty_loss(pred, target, uncertainty, alpha)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_unc_loss += unc_loss.item()
        
        return total_loss / len(train_loader), total_mse / len(train_loader), total_unc_loss / len(train_loader)
    
    def validate(self, val_loader, alpha=1.0):
        """검증"""
        self.model.eval()
        total_loss = 0
        total_mse = 0
        total_unc_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.unsqueeze(1).to(self.device)
                target = target.unsqueeze(1).to(self.device)
                
                pred, uncertainty = self.model(data)
                loss, mse_loss, unc_loss = self.uncertainty_loss(pred, target, uncertainty, alpha)
                
                total_loss += loss.item()
                total_mse += mse_loss.item()
                total_unc_loss += unc_loss.item()
        
        return total_loss / len(val_loader), total_mse / len(val_loader), total_unc_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, alpha=1.0, 
              patience=10, save_path='best_unet1d.pth'):
        """전체 학습 프로세스"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("Starting U-Net 1D training...")
        
        for epoch in range(epochs):
            # Training
            train_loss, train_mse, train_unc = self.train_epoch(train_loader, optimizer, alpha)
            
            # Validation
            val_loss, val_mse, val_unc = self.validate(val_loader, alpha)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.6f} (MSE: {train_mse:.6f}, Unc: {train_unc:.6f})')
            print(f'  Val Loss: {val_loss:.6f} (MSE: {val_mse:.6f}, Unc: {val_unc:.6f})')
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
            
            print('-' * 50)
        
        print("Training completed!")
        return self.history
    
    def predict_with_uncertainty(self, test_loader, n_samples=50):
        """Monte Carlo Dropout을 사용한 불확실성 추정과 예측"""
        self.model.eval()
        
        # Enable dropout for MC sampling
        for module in self.model.modules():
            if isinstance(module, nn.Dropout1d):
                module.train()
        
        all_predictions = []
        all_uncertainties_aleatoric = []
        all_uncertainties_epistemic = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.unsqueeze(1).to(self.device)
                target = target.to(self.device)
                
                # Monte Carlo sampling for epistemic uncertainty
                mc_predictions = []
                mc_uncertainties = []
                
                for _ in range(n_samples):
                    pred, unc = self.model(data)
                    mc_predictions.append(pred.cpu().numpy())
                    mc_uncertainties.append(unc.cpu().numpy())
                
                mc_predictions = np.array(mc_predictions)
                mc_uncertainties = np.array(mc_uncertainties)
                
                # Mean prediction (epistemic)
                mean_pred = np.mean(mc_predictions, axis=0)
                
                # Epistemic uncertainty (model uncertainty)
                epistemic_unc = np.std(mc_predictions, axis=0)
                
                # Aleatoric uncertainty (data uncertainty)
                aleatoric_unc = np.mean(mc_uncertainties, axis=0)
                
                all_predictions.append(mean_pred)
                all_uncertainties_epistemic.append(epistemic_unc)
                all_uncertainties_aleatoric.append(aleatoric_unc)
                all_targets.append(target.cpu().numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)
        epistemic_uncertainty = np.concatenate(all_uncertainties_epistemic, axis=0)
        aleatoric_uncertainty = np.concatenate(all_uncertainties_aleatoric, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Total uncertainty (combining aleatoric and epistemic)
        total_uncertainty = np.sqrt(aleatoric_uncertainty**2 + epistemic_uncertainty**2)
        
        return {
            'predictions': predictions.squeeze(),
            'targets': targets,
            'aleatoric_uncertainty': aleatoric_uncertainty.squeeze(),
            'epistemic_uncertainty': epistemic_uncertainty.squeeze(),
            'total_uncertainty': total_uncertainty.squeeze()
        }

def run_unet1d_experiment(data_dict, batch_size=32, epochs=100, learning_rate=0.001):
    """1D U-Net 실험 실행 함수"""
    
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
    
    model = UNet1D(in_channels=1, out_channels=1, dropout_p=0.1)
    trainer = UNet1DTrainer(model, device)
    
    # 학습
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=learning_rate,
        alpha=1.0,  # Uncertainty loss weight
        patience=15,
        save_path='best_unet1d.pth'
    )
    
    # 최고 모델 로드
    checkpoint = torch.load('best_unet1d.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 테스트 및 불확실성 추정
    print("\nRunning inference with uncertainty quantification...")
    results = trainer.predict_with_uncertainty(test_loader, n_samples=50)
    
    # 메트릭 계산
    predictions = results['predictions']
    targets = results['targets']
    total_uncertainty = results['total_uncertainty']
    
    metrics = calculate_metrics(targets, predictions, total_uncertainty)
    
    print("\n=== 1D U-Net Results ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")
    
    # 결과 시각화
    plot_results(
        targets, predictions, total_uncertainty, 
        title="1D U-Net Denoising Results", sample_idx=0
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

# 1D U-Net 실험 실행
unet_results = run_unet1d_experiment(
    data_dict=data_dict,
    batch_size=32,
    epochs=100,
    learning_rate=0.001
)
"""
