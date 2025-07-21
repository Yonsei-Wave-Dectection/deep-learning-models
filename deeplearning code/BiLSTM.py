import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Dict

class BiLSTMDenoiser(nn.Module):
    """BiLSTM-based Seismic Signal Denoiser with Uncertainty Quantification"""
    
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, 
                 output_size=1, dropout_p=0.2, bidirectional=True):
        super(BiLSTMDenoiser, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.bidirectional = bidirectional
        
        # Input projection layer
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # BiLSTM layers
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = hidden_size if i == 0 else hidden_size * (2 if bidirectional else 1)
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    dropout=0,  # We'll handle dropout manually
                    bidirectional=bidirectional
                )
            )
        
        # Dropout layers
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_p) for _ in range(num_layers)])
        
        # Feature dimension after BiLSTM
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size, 
            num_heads=8, 
            dropout=dropout_p,
            batch_first=True
        )
        
        # Output layers
        self.feature_projection = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
        
        # Prediction head
        self.prediction_head = nn.Linear(hidden_size // 2, output_size)
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size // 4, output_size),
            nn.Softplus()  # Ensure positive uncertainty values
        )
        
        # Variational parameters for Bayesian inference
        self.log_var = nn.Parameter(torch.zeros(1))
        
    def init_hidden(self, batch_size, device):
        """Initialize hidden states for LSTM"""
        h0_list = []
        c0_list = []
        
        for _ in range(self.num_layers):
            num_directions = 2 if self.bidirectional else 1
            h0 = torch.zeros(num_directions, batch_size, self.hidden_size).to(device)
            c0 = torch.zeros(num_directions, batch_size, self.hidden_size).to(device)
            h0_list.append(h0)
            c0_list.append(c0)
            
        return h0_list, c0_list
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        device = x.device
        
        # Reshape for processing (batch_size, seq_len, 1)
        x = x.unsqueeze(-1)
        
        # Input projection
        x = self.input_proj(x)  # (batch_size, seq_len, hidden_size)
        
        # Initialize hidden states
        h_list, c_list = self.init_hidden(batch_size, device)
        
        # Pass through BiLSTM layers
        for i, (lstm_layer, dropout_layer) in enumerate(zip(self.lstm_layers, self.dropouts)):
            # LSTM forward pass
            lstm_out, (h_list[i], c_list[i]) = lstm_layer(x, (h_list[i], c_list[i]))
            
            # Apply dropout
            x = dropout_layer(lstm_out)
        
        # Self-attention mechanism
        attn_output, attn_weights = self.attention(x, x, x)
        
        # Residual connection
        x = x + attn_output
        
        # Feature projection
        features = self.feature_projection(x)  # (batch_size, seq_len, hidden_size//2)
        
        # Generate predictions and uncertainty
        predictions = self.prediction_head(features)  # (batch_size, seq_len, 1)
        uncertainty = self.uncertainty_head(features)  # (batch_size, seq_len, 1)
        
        # Squeeze the last dimension and return
        predictions = predictions.squeeze(-1)  # (batch_size, seq_len)
        uncertainty = uncertainty.squeeze(-1)  # (batch_size, seq_len)
        
        return predictions, uncertainty, attn_weights

class VariationalBiLSTM(nn.Module):
    """Variational BiLSTM for improved uncertainty quantification"""
    
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, 
                 output_size=1, dropout_p=0.3):
        super(VariationalBiLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Variational LSTM with learnable dropout
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Variational dropout (applied during both training and inference)
        self.variational_dropout = nn.Dropout2d(p=dropout_p)
        
        lstm_output_size = hidden_size * 2  # Bidirectional
        
        # Mean prediction network
        self.mean_net = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Log variance prediction network
        self.logvar_net = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # Reshape input
        x = x.unsqueeze(-1)  # (batch_size, seq_len, 1)
        
        # BiLSTM forward pass
        lstm_out, _ = self.bilstm(x)  # (batch_size, seq_len, hidden_size*2)
        
        # Apply variational dropout (reshape for 2D dropout)
        lstm_out_2d = lstm_out.transpose(1, 2).unsqueeze(-1)  # (batch_size, hidden_size*2, seq_len, 1)
        lstm_out_2d = self.variational_dropout(lstm_out_2d)
        lstm_out = lstm_out_2d.squeeze(-1).transpose(1, 2)  # Back to (batch_size, seq_len, hidden_size*2)
        
        # Predict mean and log variance
        mean = self.mean_net(lstm_out).squeeze(-1)  # (batch_size, seq_len)
        log_var = self.logvar_net(lstm_out).squeeze(-1)  # (batch_size, seq_len)
        
        # Convert log variance to standard deviation
        std = torch.exp(0.5 * log_var)
        
        return mean, std

class BiLSTMTrainer:
    """BiLSTM 학습 클래스"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 use_variational=False):
        self.model = model.to(device)
        self.device = device
        self.use_variational = use_variational
        self.history = {'train_loss': [], 'val_loss': [], 'train_mse': [], 'val_mse': []}
    
    def gaussian_nll_loss(self, pred_mean, pred_std, target):
        """Gaussian Negative Log-Likelihood Loss"""
        var = pred_std ** 2
        loss = 0.5 * (torch.log(2 * np.pi * var) + (target - pred_mean) ** 2 / var)
        return torch.mean(loss)
    
    def heteroscedastic_loss(self, pred, target, uncertainty, alpha=1.0):
        """Heteroscedastic uncertainty loss"""
        precision = 1.0 / (uncertainty + 1e-8)
        mse_loss = torch.mean(precision * (pred - target) ** 2)
        uncertainty_regularization = torch.mean(torch.log(uncertainty + 1e-8))
        
        total_loss = mse_loss + alpha * uncertainty_regularization
        return total_loss, mse_loss, uncertainty_regularization
    
    def kl_divergence_loss(self, model):
        """KL divergence regularization for Bayesian networks"""
        kl_loss = 0
        for module in model.modules():
            if hasattr(module, 'log_var'):
                # Simple KL divergence for Gaussian prior
                kl_loss += -0.5 * torch.sum(1 + module.log_var - module.log_var.exp())
        return kl_loss
    
    def train_epoch(self, train_loader, optimizer, alpha=1.0, beta=0.001):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0
        total_mse = 0
        total_unc_loss = 0
        total_kl_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            
            optimizer.zero_grad()
            
            if self.use_variational:
                # Variational BiLSTM
                pred_mean, pred_std = self.model(data)
                loss = self.gaussian_nll_loss(pred_mean, pred_std, target)
                mse_loss = F.mse_loss(pred_mean, target)
                unc_loss = torch.zeros_like(loss)
                kl_loss = torch.zeros_like(loss)
            else:
                # Standard BiLSTM with uncertainty
                pred, uncertainty, _ = self.model(data)
                loss, mse_loss, unc_loss = self.heteroscedastic_loss(pred, target, uncertainty, alpha)
                kl_loss = self.kl_divergence_loss(self.model)
                loss += beta * kl_loss
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_unc_loss += unc_loss.item() if not self.use_variational else 0
            total_kl_loss += kl_loss.item()
        
        return (total_loss / len(train_loader), 
                total_mse / len(train_loader), 
                total_unc_loss / len(train_loader),
                total_kl_loss / len(train_loader))
    
    def validate(self, val_loader, alpha=1.0):
        """검증"""
        self.model.eval()
        total_loss = 0
        total_mse = 0
        total_unc_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                
                if self.use_variational:
                    pred_mean, pred_std = self.model(data)
                    loss = self.gaussian_nll_loss(pred_mean, pred_std, target)
                    mse_loss = F.mse_loss(pred_mean, target)
                    unc_loss = torch.zeros_like(loss)
                else:
                    pred, uncertainty, _ = self.model(data)
                    loss, mse_loss, unc_loss = self.heteroscedastic_loss(pred, target, uncertainty, alpha)
                
                total_loss += loss.item()
                total_mse += mse_loss.item()
                total_unc_loss += unc_loss.item() if not self.use_variational else 0
        
        return (total_loss / len(val_loader), 
                total_mse / len(val_loader), 
                total_unc_loss / len(val_loader))
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, alpha=1.0, beta=0.001,
              patience=15, save_path='best_bilstm.pth'):
        """전체 학습 프로세스"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.7)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("Starting BiLSTM training...")
        
        for epoch in range(epochs):
            # Training
            train_loss, train_mse, train_unc, train_kl = self.train_epoch(
                train_loader, optimizer, alpha, beta
            )
            
            # Validation
            val_loss, val_mse, val_unc = self.validate(val_loader, alpha)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mse'].append(train_mse)
            self.history['val_mse'].append(val_mse)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.6f} (MSE: {train_mse:.6f}, Unc: {train_unc:.6f}, KL: {train_kl:.6f})')
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
        
        if self.use_variational:
            # Variational inference
            self.model.eval()
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(self.device)
                    target = target.to(self.device)
                    
                    pred_mean, pred_std = self.model(data)
                    
                    predictions.append(pred_mean.cpu().numpy())
                    uncertainties.append(pred_std.cpu().numpy())
                    epistemic_uncertainties.append(np.zeros_like(pred_std.cpu().numpy()))
                    targets.append(target.cpu().numpy())
        else:
            # Monte Carlo Dropout for epistemic uncertainty
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    module.train()  # Enable dropout during inference
            
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(self.device)
                    target = target.to(self.device)
                    
                    # Multiple forward passes for MC Dropout
                    mc_predictions = []
                    mc_uncertainties = []
                    mc_attentions = []
                    
                    for _ in range(n_samples):
                        pred, unc, attn = self.model(data)
                        mc_predictions.append(pred.cpu().numpy())
                        mc_uncertainties.append(unc.cpu().numpy())
                        mc_attentions.append(attn.cpu().numpy())
                    
                    mc_predictions = np.array(mc_predictions)
                    mc_uncertainties = np.array(mc_uncertainties)
                    
                    # Mean prediction
                    mean_pred = np.mean(mc_predictions, axis=0)
                    
                    # Epistemic uncertainty (model uncertainty)
                    epistemic_unc = np.std(mc_predictions, axis=0)
                    
                    # Aleatoric uncertainty (average predicted uncertainty)
                    aleatoric_unc = np.mean(mc_uncertainties, axis=0)
                    
                    predictions.append(mean_pred)
                    uncertainties.append(aleatoric_unc)
                    epistemic_uncertainties.append(epistemic_unc)
                    targets.append(target.cpu().numpy())
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
            'attention_weights': attention_weights if not self.use_variational else None
        }

def run_bilstm_experiment(data_dict, model_type='standard', batch_size=32, 
                         epochs=100, learning_rate=0.001):
    """BiLSTM 실험 실행 함수"""
    
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
    
    if model_type == 'variational':
        model = VariationalBiLSTM(
            input_size=1, 
            hidden_size=128, 
            num_layers=3, 
            dropout_p=0.3
        )
        use_variational = True
    else:
        model = BiLSTMDenoiser(
            input_size=1, 
            hidden_size=128, 
            num_layers=3, 
            dropout_p=0.2
        )
        use_variational = False
    
    trainer = BiLSTMTrainer(model, device, use_variational=use_variational)
    
    # 학습
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=learning_rate,
        alpha=1.0,  # Uncertainty loss weight
        beta=0.001,  # KL divergence weight
        patience=15,
        save_path=f'best_bilstm_{model_type}.pth'
    )
    
    # 최고 모델 로드
    checkpoint = torch.load(f'best_bilstm_{model_type}.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 테스트 및 불확실성 추정
    print("\nRunning inference with uncertainty quantification...")
    results = trainer.predict_with_uncertainty(test_loader, n_samples=50)
    
    # 메트릭 계산
    predictions = results['predictions']
    targets = results['targets']
    total_uncertainty = results['total_uncertainty']
    
    metrics = calculate_metrics(targets, predictions, total_uncertainty)
    
    print(f"\n=== BiLSTM ({model_type}) Results ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")
    
    # 결과 시각화
    plot_results(
        targets, predictions, total_uncertainty, 
        title=f"BiLSTM ({model_type}) Denoising Results", sample_idx=0
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

# Standard BiLSTM 실험
bilstm_results = run_bilstm_experiment(
    data_dict=data_dict,
    model_type='standard',
    batch_size=32,
    epochs=100,
    learning_rate=0.001
)

# Variational BiLSTM 실험
var_bilstm_results = run_bilstm_experiment(
    data_dict=data_dict,
    model_type='variational',
    batch_size=32,
    epochs=100,
    learning_rate=0.001
)
"""
