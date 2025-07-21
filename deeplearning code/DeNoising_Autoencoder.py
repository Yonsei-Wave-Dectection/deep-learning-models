import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Dict

class ResidualBlock1D(nn.Module):
    """1D Residual Block for better gradient flow"""
    def __init__(self, channels, kernel_size=3, dropout_p=0.1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout1d(dropout_p)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class DenoisingAutoencoder(nn.Module):
    """Deep Denoising Autoencoder with Uncertainty Quantification"""
    
    def __init__(self, input_length=1000, encoding_dims=[64, 128, 256, 512], 
                 latent_dim=256, dropout_p=0.2):
        super(DenoisingAutoencoder, self).__init__()
        
        self.input_length = input_length
        self.encoding_dims = encoding_dims
        self.latent_dim = latent_dim
        self.dropout_p = dropout_p
        
        # Encoder
        self.encoder = nn.Sequential()
        in_channels = 1
        current_length = input_length
        
        for i, dim in enumerate(encoding_dims):
            self.encoder.add_module(f'conv{i}', nn.Conv1d(in_channels, dim, 5, stride=2, padding=2))
            self.encoder.add_module(f'bn{i}', nn.BatchNorm1d(dim))
            self.encoder.add_module(f'relu{i}', nn.ReLU(inplace=True))
            self.encoder.add_module(f'dropout{i}', nn.Dropout1d(dropout_p))
            self.encoder.add_module(f'res{i}', ResidualBlock1D(dim, dropout_p=dropout_p))
            in_channels = dim
            current_length = (current_length + 1) // 2
        
        self.encoded_length = current_length
        
        # Latent space
        self.encode_fc = nn.Linear(encoding_dims[-1] * self.encoded_length, latent_dim)
        self.decode_fc = nn.Linear(latent_dim, encoding_dims[-1] * self.encoded_length)
        
        # Main decoder (for reconstruction)
        self.decoder = nn.Sequential()
        decoding_dims = encoding_dims[::-1]
        
        for i, (in_dim, out_dim) in enumerate(zip(decoding_dims, decoding_dims[1:] + [1])):
            self.decoder.add_module(f'res_dec{i}', ResidualBlock1D(in_dim, dropout_p=dropout_p))
            self.decoder.add_module(f'deconv{i}', 
                nn.ConvTranspose1d(in_dim, out_dim, 5, stride=2, padding=2, output_padding=1))
            if out_dim != 1:
                self.decoder.add_module(f'bn_dec{i}', nn.BatchNorm1d(out_dim))
                self.decoder.add_module(f'relu_dec{i}', nn.ReLU(inplace=True))
                self.decoder.add_module(f'dropout_dec{i}', nn.Dropout1d(dropout_p))
        
        # Uncertainty decoder (parallel branch)
        self.uncertainty_decoder = nn.Sequential()
        
        for i, (in_dim, out_dim) in enumerate(zip(decoding_dims, decoding_dims[1:] + [1])):
            self.uncertainty_decoder.add_module(f'res_unc{i}', ResidualBlock1D(in_dim, dropout_p=dropout_p))
            self.uncertainty_decoder.add_module(f'deconv_unc{i}', 
                nn.ConvTranspose1d(in_dim, out_dim, 5, stride=2, padding=2, output_padding=1))
            if out_dim != 1:
                self.uncertainty_decoder.add_module(f'bn_unc{i}', nn.BatchNorm1d(out_dim))
                self.uncertainty_decoder.add_module(f'relu_unc{i}', nn.ReLU(inplace=True))
                self.uncertainty_decoder.add_module(f'dropout_unc{i}', nn.Dropout1d(dropout_p))
            else:
                self.uncertainty_decoder.add_module(f'softplus_unc{i}', nn.Softplus())
    
    def encode(self, x):
        """Encode input to latent space"""
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        latent = self.encode_fc(x)
        return latent
    
    def decode(self, latent):
        """Decode from latent space to reconstruction"""
        x = self.decode_fc(latent)
        x = x.view(x.size(0), self.encoding_dims[-1], self.encoded_length)
        reconstruction = self.decoder(x)
        return reconstruction.squeeze(1)
    
    def decode_uncertainty(self, latent):
        """Decode from latent space to uncertainty"""
        x = self.decode_fc(latent)
        x = x.view(x.size(0), self.encoding_dims[-1], self.encoded_length)
        uncertainty = self.uncertainty_decoder(x)
        return uncertainty.squeeze(1)
    
    def forward(self, x):
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        uncertainty = self.decode_uncertainty(latent)
        return reconstruction, uncertainty, latent

class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder for better uncertainty quantification"""
    
    def __init__(self, input_length=1000, encoding_dims=[64, 128, 256], 
                 latent_dim=128, dropout_p=0.2):
        super(VariationalAutoencoder, self).__init__()
        
        self.input_length = input_length
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential()
        in_channels = 1
        current_length = input_length
        
        for i, dim in enumerate(encoding_dims):
            self.encoder.add_module(f'conv{i}', nn.Conv1d(in_channels, dim, 5, stride=2, padding=2))
            self.encoder.add_module(f'bn{i}', nn.BatchNorm1d(dim))
            self.encoder.add_module(f'relu{i}', nn.ReLU(inplace=True))
            self.encoder.add_module(f'dropout{i}', nn.Dropout1d(dropout_p))
            in_channels = dim
            current_length = (current_length + 1) // 2
        
        self.encoded_length = current_length
        final_size = encoding_dims[-1] * self.encoded_length
        
        # Latent space (mean and log variance)
        self.fc_mu = nn.Linear(final_size, latent_dim)
        self.fc_logvar = nn.Linear(final_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, final_size)
        
        # Decoder
        self.decoder = nn.Sequential()
        decoding_dims = encoding_dims[::-1]
        
        for i, (in_dim, out_dim) in enumerate(zip(decoding_dims, decoding_dims[1:] + [1])):
            self.decoder.add_module(f'deconv{i}', 
                nn.ConvTranspose1d(in_dim, out_dim, 5, stride=2, padding=2, output_padding=1))
            if out_dim != 1:
                self.decoder.add_module(f'bn{i}', nn.BatchNorm1d(out_dim))
                self.decoder.add_module(f'relu{i}', nn.ReLU(inplace=True))
                self.decoder.add_module(f'dropout{i}', nn.Dropout1d(dropout_p))
    
    def encode(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(x.size(0), -1, self.encoded_length)
        x = self.decoder(x)
        return x.squeeze(1)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

class DenoisingAETrainer:
    """Denoising Autoencoder 학습 클래스"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 model_type='standard'):
        self.model = model.to(device)
        self.device = device
        self.model_type = model_type
        self.history = {'train_loss': [], 'val_loss': [], 'train_recon': [], 'val_recon': []}
    
    def vae_loss(self, recon, target, mu, logvar, beta=1.0):
        """VAE loss with KL divergence"""
        recon_loss = F.mse_loss(recon, target, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl_loss, recon_loss, kl_loss
    
    def uncertainty_loss(self, recon, target, uncertainty, alpha=1.0):
        """Uncertainty-aware loss"""
        precision = 1.0 / (uncertainty + 1e-8)
        recon_loss = torch.mean(precision * (recon - target) ** 2)
        uncertainty_reg = torch.mean(torch.log(uncertainty + 1e-8))
        return recon_loss + alpha * uncertainty_reg, recon_loss, uncertainty_reg
    
    def train_epoch(self, train_loader, optimizer, alpha=1.0, beta=1.0):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_reg = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            
            optimizer.zero_grad()
            
            if self.model_type == 'vae':
                recon, mu, logvar = self.model(data)
                loss, recon_loss, kl_loss = self.vae_loss(recon, target, mu, logvar, beta)
                reg_loss = kl_loss
            else:
                recon, uncertainty, _ = self.model(data)
                loss, recon_loss, reg_loss = self.uncertainty_loss(recon, target, uncertainty, alpha)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_reg += reg_loss.item()
        
        return (total_loss / len(train_loader), 
                total_recon / len(train_loader), 
                total_reg / len(train_loader))
    
    def validate(self, val_loader, alpha=1.0, beta=1.0):
        """검증"""
        self.model.eval()
        total_loss = 0
        total_recon = 0
        total_reg = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                
                if self.model_type == 'vae':
                    recon, mu, logvar = self.model(data)
                    loss, recon_loss, kl_loss = self.vae_loss(recon, target, mu, logvar, beta)
                    reg_loss = kl_loss
                else:
                    recon, uncertainty, _ = self.model(data)
                    loss, recon_loss, reg_loss = self.uncertainty_loss(recon, target, uncertainty, alpha)
                
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_reg += reg_loss.item()
        
        return (total_loss / len(val_loader), 
                total_recon / len(val_loader), 
                total_reg / len(val_loader))
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, alpha=1.0, beta=1.0,
              patience=15, save_path='best_denoising_ae.pth'):
        """전체 학습 프로세스"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.7)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting Denoising Autoencoder ({self.model_type}) training...")
        
        for epoch in range(epochs):
            # Training
            train_loss, train_recon, train_reg = self.train_epoch(train_loader, optimizer, alpha, beta)
            
            # Validation
            val_loss, val_recon, val_reg = self.validate(val_loader, alpha, beta)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_recon'].append(train_recon)
            self.history['val_recon'].append(val_recon)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.6f} (Recon: {train_recon:.6f}, Reg: {train_reg:.6f})')
            print(f'  Val Loss: {val_loss:.6f} (Recon: {val_recon:.6f}, Reg: {val_reg:.6f})')
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
        latent_representations = []
        
        if self.model_type == 'vae':
            # VAE inference with multiple sampling
            self.model.eval()
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(self.device)
                    target = target.to(self.device)
                    
                    # Multiple samples for epistemic uncertainty
                    sample_predictions = []
                    for _ in range(n_samples):
                        mu, logvar = self.model.encode(data)
                        z = self.model.reparameterize(mu, logvar)
                        recon = self.model.decode(z)
                        sample_predictions.append(recon.cpu().numpy())
                    
                    sample_predictions = np.array(sample_predictions)
                    mean_pred = np.mean(sample_predictions, axis=0)
                    epistemic_unc = np.std(sample_predictions, axis=0)
                    
                    predictions.append(mean_pred)
                    uncertainties.append(epistemic_unc)  # For VAE, epistemic = total
                    epistemic_uncertainties.append(epistemic_unc)
                    targets.append(target.cpu().numpy())
        else:
            # Standard autoencoder with MC Dropout
            for module in self.model.modules():
                if isinstance(module, nn.Dropout1d):
                    module.train()
            
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(self.device)
                    target = target.to(self.device)
                    
                    # MC Dropout sampling
                    mc_predictions = []
                    mc_uncertainties = []
                    mc_latents = []
                    
                    for _ in range(n_samples):
                        recon, unc, latent = self.model(data)
                        mc_predictions.append(recon.cpu().numpy())
                        mc_uncertainties.append(unc.cpu().numpy())
                        mc_latents.append(latent.cpu().numpy())
                    
                    mc_predictions = np.array(mc_predictions)
                    mc_uncertainties = np.array(mc_uncertainties)
                    
                    # Mean prediction and uncertainties
                    mean_pred = np.mean(mc_predictions, axis=0)
                    epistemic_unc = np.std(mc_predictions, axis=0)
                    aleatoric_unc = np.mean(mc_uncertainties, axis=0)
                    
                    predictions.append(mean_pred)
                    uncertainties.append(aleatoric_unc)
                    epistemic_uncertainties.append(epistemic_unc)
                    targets.append(target.cpu().numpy())
                    latent_representations.append(np.mean(mc_latents, axis=0))
        
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
            'latent_representations': latent_representations if latent_representations else None
        }

def run_denoising_ae_experiment(data_dict, model_type='standard', batch_size=32, 
                               epochs=100, learning_rate=0.001):
    """Denoising Autoencoder 실험 실행 함수"""
    
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
    
    input_length = data_dict['X_train'].shape[1]
    
    if model_type == 'vae':
        model = VariationalAutoencoder(
            input_length=input_length,
            encoding_dims=[64, 128, 256],
            latent_dim=128,
            dropout_p=0.2
        )
    else:
        model = DenoisingAutoencoder(
            input_length=input_length,
            encoding_dims=[64, 128, 256, 512],
            latent_dim=256,
            dropout_p=0.2
        )
    
    trainer = DenoisingAETrainer(model, device, model_type=model_type)
    
    # 학습
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=learning_rate,
        alpha=1.0,  # Uncertainty loss weight
        beta=0.1,   # KL divergence weight for VAE
        patience=15,
        save_path=f'best_denoising_ae_{model_type}.pth'
    )
    
    # 최고 모델 로드
    checkpoint = torch.load(f'best_denoising_ae_{model_type}.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 테스트 및 불확실성 추정
    print("\nRunning inference with uncertainty quantification...")
    results = trainer.predict_with_uncertainty(test_loader, n_samples=50)
    
    # 메트릭 계산
    predictions = results['predictions']
    targets = results['targets']
    total_uncertainty = results['total_uncertainty']
    
    metrics = calculate_metrics(targets, predictions, total_uncertainty)
    
    print(f"\n=== Denoising Autoencoder ({model_type}) Results ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")
    
    # 결과 시각화
    plot_results(
        targets, predictions, total_uncertainty, 
        title=f"Denoising AE ({model_type}) Results", sample_idx=0
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

# Standard Denoising Autoencoder 실험
dae_results = run_denoising_ae_experiment(
    data_dict=data_dict,
    model_type='standard',
    batch_size=32,
    epochs=100,
    learning_rate=0.001
)

# Variational Autoencoder 실험
vae_results = run_denoising_ae_experiment(
    data_dict=data_dict,
    model_type='vae',
    batch_size=32,
    epochs=100,
    learning_rate=0.001
)
"""
