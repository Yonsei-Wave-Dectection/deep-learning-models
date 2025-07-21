import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import math
from typing import Tuple, Dict

class AutoCorrelation(nn.Module):
    """Auto-Correlation mechanism for Autoformer"""
    
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
    
    def time_delay_agg_training(self, values, corr):
        """Time delay aggregation for training"""
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        
        # Find top k correlations
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        
        # Aggregate values based on top correlations
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
        
        return delays_agg
    
    def time_delay_agg_inference(self, values, corr):
        """Time delay aggregation for inference"""
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        
        # Speed up computation
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        
        # Aggregate
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
        
        return delays_agg
    
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]
        
        # Compute auto-correlation
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        
        # Time delay aggregation
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        
        return V.contiguous(), corr.permute(0, 3, 1, 2)

class AutoCorrelationLayer(nn.Module):
    """Auto-correlation layer with multi-head mechanism"""
    
    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None, dropout=0.1):
        super(AutoCorrelationLayer, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        
        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        
        out, attn = self.inner_correlation(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        
        return self.dropout(self.out_projection(out)), attn

class SeriesDecomposition(nn.Module):
    """Series decomposition block for trend and seasonal components"""
    
    def __init__(self, kernel_size):
        super(SeriesDecomposition, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)
    
    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class MovingAvg(nn.Module):
    """Moving average block to highlight the trend of time series"""
    
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        # Padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class AutoformerLayer(nn.Module):
    """Autoformer layer with auto-correlation and series decomposition"""
    
    def __init__(self, autocorrelation, d_model, n_heads, d_ff=None, 
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(AutoformerLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        self.autocorrelation = autocorrelation
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.decomp1 = SeriesDecomposition(moving_avg)
        self.decomp2 = SeriesDecomposition(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
    
    def forward(self, x, attn_mask=None):
        new_x, attn = self.autocorrelation(x, x, x, attn_mask)
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        
        return res, attn

class Autoformer(nn.Module):
    """Autoformer for seismic signal denoising with uncertainty quantification"""
    
    def __init__(self, seq_len=1000, d_model=512, n_heads=8, e_layers=2, d_layers=1,
                 d_ff=2048, moving_avg=25, factor=1, dropout=0.1, activation='gelu'):
        super(Autoformer, self).__init__()
        
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Input embedding
        self.enc_embedding = DataEmbedding(1, d_model, dropout)
        self.dec_embedding = DataEmbedding(1, d_model, dropout)
        
        # Encoder
        self.encoder = Encoder(
            [
                AutoformerLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout),
                        d_model, n_heads
                    ),
                    d_model, n_heads, d_ff, moving_avg, dropout, activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # Decoder
        self.decoder = Decoder(
            [
                AutoformerLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=dropout),
                        d_model, n_heads
                    ),
                    d_model, n_heads, d_ff, moving_avg, dropout, activation
                ) for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # Final projection layers
        self.projection = nn.Linear(d_model, 1)
        
        # Uncertainty estimation
        self.uncertainty_layers = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
            nn.Softplus()
        )
        
        # Trend and seasonal decomposition for uncertainty
        self.trend_decomp = SeriesDecomposition(moving_avg)
        
    def forward(self, x):
        # x: (batch_size, seq_len)
        x = x.unsqueeze(-1)  # (batch_size, seq_len, 1)
        
        # Embedding
        enc_out = self.enc_embedding(x)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        # Use encoder output as decoder input for denoising task
        dec_out = self.dec_embedding(x)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        
        # Final prediction
        prediction = self.projection(dec_out).squeeze(-1)
        
        # Uncertainty estimation
        uncertainty = self.uncertainty_layers(dec_out).squeeze(-1)
        
        # Trend-seasonal decomposition for additional uncertainty info
        seasonal, trend = self.trend_decomp(dec_out)
        
        return prediction, uncertainty, {'trend': trend, 'seasonal': seasonal, 'attentions': attns}

class DataEmbedding(nn.Module):
    """Data embedding with positional encoding"""
    
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)

class TokenEmbedding(nn.Module):
    """Token embedding using 1D convolution"""
    
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                  kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
    
    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class PositionalEmbedding(nn.Module):
    """Positional embedding"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return self.pe[:, :x.size(1)]

class Encoder(nn.Module):
    """Autoformer encoder"""
    
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
    
    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x, attns

class Decoder(nn.Module):
    """Autoformer decoder"""
    
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
    
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x, _ = layer(x, attn_mask=x_mask)
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x

class AutoformerTrainer:
    """Autoformer 학습 클래스"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {'train_loss': [], 'val_loss': [], 'train_mse': [], 'val_mse': []}
    
    def autoformer_loss(self, pred, target, uncertainty, extra_info, alpha=1.0, beta=0.1):
        """Autoformer-specific loss with trend-seasonal decomposition"""
        # Main reconstruction loss
        mse_loss = F.mse_loss(pred, target)
        
        # Uncertainty-aware loss
        precision = 1.0 / (uncertainty + 1e-8)
        uncertainty_loss = torch.mean(precision * (pred - target) ** 2)
        uncertainty_reg = torch.mean(torch.log(uncertainty + 1e-8))
        
        # Trend preservation loss
        if 'trend' in extra_info:
            pred_trend = extra_info['trend'].mean(dim=-1)  # Average over features
            target_trend = self.extract_trend(target)
            trend_loss = F.mse_loss(pred_trend, target_trend)
        else:
            trend_loss = torch.tensor(0.0).to(pred.device)
        
        # Seasonal pattern loss
        if 'seasonal' in extra_info:
            seasonal_loss = torch.mean(torch.abs(extra_info['seasonal']))
        else:
            seasonal_loss = torch.tensor(0.0).to(pred.device)
        
        total_loss = uncertainty_loss + alpha * uncertainty_reg + \
                    beta * trend_loss + 0.05 * seasonal_loss
        
        return total_loss, mse_loss, uncertainty_reg, trend_loss, seasonal_loss
    
    def extract_trend(self, x, kernel_size=25):
        """Extract trend using moving average"""
        # Simple moving average for trend extraction
        padding = (kernel_size - 1) // 2
        x_padded = F.pad(x, (padding, padding), mode='reflect')
        trend = F.avg_pool1d(x_padded.unsqueeze(1), kernel_size=kernel_size, stride=1).squeeze(1)
        return trend
    
    def train_epoch(self, train_loader, optimizer, alpha=1.0, beta=0.1):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0
        total_mse = 0
        total_unc_reg = 0
        total_trend_loss = 0
        total_seasonal_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            
            optimizer.zero_grad()
            
            pred, uncertainty, extra_info = self.model(data)
            loss, mse_loss, unc_reg, trend_loss, seasonal_loss = self.autoformer_loss(
                pred, target, uncertainty, extra_info, alpha, beta
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_unc_reg += unc_reg.item()
            total_trend_loss += trend_loss.item()
            total_seasonal_loss += seasonal_loss.item()
        
        return (total_loss / len(train_loader), 
                total_mse / len(train_loader),
                total_unc_reg / len(train_loader),
                total_trend_loss / len(train_loader),
                total_seasonal_loss / len(train_loader))
    
    def validate(self, val_loader, alpha=1.0, beta=0.1):
        """검증"""
        self.model.eval()
        total_loss = 0
        total_mse = 0
        total_unc_reg = 0
        total_trend_loss = 0
        total_seasonal_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                
                pred, uncertainty, extra_info = self.model(data)
                loss, mse_loss, unc_reg, trend_loss, seasonal_loss = self.autoformer_loss(
                    pred, target, uncertainty, extra_info, alpha, beta
                )
                
                total_loss += loss.item()
                total_mse += mse_loss.item()
                total_unc_reg += unc_reg.item()
                total_trend_loss += trend_loss.item()
                total_seasonal_loss += seasonal_loss.item()
        
        return (total_loss / len(val_loader), 
                total_mse / len(val_loader),
                total_unc_reg / len(val_loader),
                total_trend_loss / len(val_loader),
                total_seasonal_loss / len(val_loader))
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.0001, alpha=1.0, beta=0.1,
              patience=15, save_path='best_autoformer.pth'):
        """전체 학습 프로세스"""
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.1)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("Starting Autoformer training...")
        
        for epoch in range(epochs):
            # Training
            train_loss, train_mse, train_unc, train_trend, train_seasonal = self.train_epoch(
                train_loader, optimizer, alpha, beta
            )
            
            # Validation
            val_loss, val_mse, val_unc, val_trend, val_seasonal = self.validate(
                val_loader, alpha, beta
            )
            
            # Learning rate scheduling
            scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mse'].append(train_mse)
            self.history['val_mse'].append(val_mse)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.6f} (MSE: {train_mse:.6f}, Unc: {train_unc:.6f}, Trend: {train_trend:.6f}, Seasonal: {train_seasonal:.6f})')
            print(f'  Val Loss: {val_loss:.6f} (MSE: {val_mse:.6f}, Unc: {val_unc:.6f}, Trend: {val_trend:.6f}, Seasonal: {val_seasonal:.6f})')
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
        trends = []
        seasonals = []
        
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
                mc_trends = []
                mc_seasonals = []
                
                for _ in range(n_samples):
                    pred, unc, extra_info = self.model(data)
                    mc_predictions.append(pred.cpu().numpy())
                    mc_uncertainties.append(unc.cpu().numpy())
                    if 'trend' in extra_info:
                        mc_trends.append(extra_info['trend'].mean(dim=-1).cpu().numpy())
                    if 'seasonal' in extra_info:
                        mc_seasonals.append(extra_info['seasonal'].mean(dim=-1).cpu().numpy())
                
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
                
                if mc_trends:
                    trends.append(np.mean(mc_trends, axis=0))
                if mc_seasonals:
                    seasonals.append(np.mean(mc_seasonals, axis=0))
        
        # Concatenate results
        predictions = np.concatenate(predictions, axis=0)
        aleatoric_uncertainty = np.concatenate(uncertainties, axis=0)
        epistemic_uncertainty = np.concatenate(epistemic_uncertainties, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        # Total uncertainty
        total_uncertainty = np.sqrt(aleatoric_uncertainty**2 + epistemic_uncertainty**2)
        
        result = {
            'predictions': predictions,
            'targets': targets,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'epistemic_uncertainty': epistemic_uncertainty,
            'total_uncertainty': total_uncertainty
        }
        
        if trends:
            result['trends'] = np.concatenate(trends, axis=0)
        if seasonals:
            result['seasonals'] = np.concatenate(seasonals, axis=0)
        
        return result

def run_autoformer_experiment(data_dict, batch_size=32, epochs=100, learning_rate=0.0001):
    """Autoformer 실험 실행 함수"""
    
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
    
    model = Autoformer(
        seq_len=seq_len,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        moving_avg=25,
        factor=1,
        dropout=0.1,
        activation='gelu'
    )
    
    trainer = AutoformerTrainer(model, device)
    
    # 학습
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=learning_rate,
        alpha=1.0,  # Uncertainty regularization weight
        beta=0.1,   # Trend loss weight
        patience=15,
        save_path='best_autoformer.pth'
    )
    
    # 최고 모델 로드
    checkpoint = torch.load('best_autoformer.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 테스트 및 불확실성 추정
    print("\nRunning inference with uncertainty quantification...")
    results = trainer.predict_with_uncertainty(test_loader, n_samples=50)
    
    # 메트릭 계산
    predictions = results['predictions']
    targets = results['targets']
    total_uncertainty = results['total_uncertainty']
    
    metrics = calculate_metrics(targets, predictions, total_uncertainty)
    
    print("\n=== Autoformer Results ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")
    
    # 결과 시각화
    plot_results(
        targets, predictions, total_uncertainty, 
        title="Autoformer Denoising Results", sample_idx=0
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

# Autoformer 실험
autoformer_results = run_autoformer_experiment(
    data_dict=data_dict,
    batch_size=16,  # Autoformer는 메모리를 많이 사용하므로 배치 크기를 줄임
    epochs=100,
    learning_rate=0.0001
)
"""
