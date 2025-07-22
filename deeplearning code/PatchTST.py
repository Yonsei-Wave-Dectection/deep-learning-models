import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import math
from typing import Tuple, Dict

class PatchEmbedding(nn.Module):
    """Patch embedding for time series"""
    
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        
        self.patch_len = patch_len
        self.stride = stride
        
        # Patch projection
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        
        # Positional embedding
        self.position_embedding = nn.Parameter(torch.randn(1024, d_model) * 0.02)  # Max 1024 patches
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch_size, seq_len, n_vars) or (batch_size, seq_len) for univariate
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)  # Add variable dimension
        
        batch_size, seq_len, n_vars = x.shape
        
        # Create patches
        num_patches = (seq_len - self.patch_len) // self.stride + 1
        patches = []
        
        for i in range(num_patches):
            start_idx = i * self.stride
            end_idx = start_idx + self.patch_len
            patch = x[:, start_idx:end_idx, :]  # (batch_size, patch_len, n_vars)
            patches.append(patch)
        
        patches = torch.stack(patches, dim=1)  # (batch_size, num_patches, patch_len, n_vars)
        
        # Flatten patches for embedding
        patches = patches.reshape(batch_size, num_patches, -1)  # (batch_size, num_patches, patch_len * n_vars)
        
        # Apply patch embedding
        patch_embeddings = self.value_embedding(patches)  # (batch_size, num_patches, d_model)
        
        # Add positional embeddings
        pos_embeddings = self.position_embedding[:num_patches, :].unsqueeze(0)
        patch_embeddings = patch_embeddings + pos_embeddings
        
        return self.dropout(patch_embeddings), num_patches

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
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

class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model, d_ff, dropout=0.1, activation='gelu'):
        super(FeedForward, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, activation='gelu'):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, attn_weights = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x, attn_weights

class PatchTST(nn.Module):
    """PatchTST for seismic signal denoising with uncertainty quantification"""
    
    def __init__(self, seq_len=1000, patch_len=16, stride=8, d_model=512, n_heads=8, 
                 n_layers=3, d_ff=2048, dropout=0.1, activation='gelu'):
        super(PatchTST, self).__init__()
        
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, dropout)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, activation)
            for _ in range(n_layers)
        ])
        
        # Calculate number of patches
        self.num_patches = (seq_len - patch_len) // stride + 1
        
        # Head for reconstruction
        self.head = nn.Linear(d_model, patch_len)
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, patch_len),
            nn.Softplus()
        )
        
        # Global uncertainty context
        self.global_uncertainty = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1), 
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        batch_size, seq_len = x.shape
        
        # Patch embedding
        patch_embeddings, num_patches = self.patch_embedding(x)  # (batch_size, num_patches, d_model)
        
        # Store attention weights
        attention_weights = []
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            patch_embeddings, attn_weights = layer(patch_embeddings)
            attention_weights.append(attn_weights)
        
        # Generate patch predictions
        patch_predictions = self.head(patch_embeddings)  # (batch_size, num_patches, patch_len)
        
        # Generate patch uncertainties
        patch_uncertainties = self.uncertainty_head(patch_embeddings)  # (batch_size, num_patches, patch_len)
        
        # Global uncertainty context
        global_context = self.global_uncertainty(patch_embeddings.transpose(1, 2))  # (batch_size, 1, 1)
        global_context = global_context.unsqueeze(2)  # (batch_size, 1, 1)
        
        # Apply global context to uncertainties
        patch_uncertainties = patch_uncertainties * global_context
        
        # Reconstruct full sequence from patches
        predictions = self.reconstruct_from_patches(patch_predictions, seq_len)
        uncertainties = self.reconstruct_from_patches(patch_uncertainties, seq_len)
        
        return predictions, uncertainties, attention_weights
    
    def reconstruct_from_patches(self, patches, target_len):
        """Reconstruct full sequence from patches with overlap handling"""
        batch_size, num_patches, patch_len = patches.shape
        
        # Initialize reconstruction tensor
        reconstruction = torch.zeros(batch_size, target_len, device=patches.device)
        counts = torch.zeros(batch_size, target_len, device=patches.device)
        
        # Add patches to reconstruction
        for i in range(num_patches):
            start_idx = i * self.stride
            end_idx = start_idx + patch_len
            
            if end_idx <= target_len:
                reconstruction[:, start_idx:end_idx] += patches[:, i, :]
                counts[:, start_idx:end_idx] += 1
        
        # Average overlapping regions
        reconstruction = reconstruction / (counts + 1e-8)
        
        return reconstruction

class MultiscalePatchTST(nn.Module):
    """Multi-scale PatchTST with different patch sizes"""
    
    def __init__(self, seq_len=1000, patch_lens=[8, 16, 32], strides=None, 
                 d_model=512, n_heads=8, n_layers=3, d_ff=2048, dropout=0.1):
        super(MultiscalePatchTST, self).__init__()
        
        self.seq_len = seq_len
        self.patch_lens = patch_lens
        self.strides = strides or [p // 2 for p in patch_lens]  # Default stride = patch_len // 2
        
        # Multi-scale patch embeddings
        self.patch_embeddings = nn.ModuleList([
            PatchEmbedding(d_model, patch_len, stride, dropout)
            for patch_len, stride in zip(patch_lens, self.strides)
        ])
        
        # Shared transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Scale-specific heads
        self.heads = nn.ModuleList([
            nn.Linear(d_model, patch_len)
            for patch_len in patch_lens
        ])
        
        # Scale-specific uncertainty heads
        self.uncertainty_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, patch_len),
                nn.Softplus()
            )
            for patch_len in patch_lens
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(len(patch_lens), len(patch_lens)),
            nn.Softmax(dim=-1)
        )
        
        # Global uncertainty
        self.global_uncertainty = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        scale_predictions = []
        scale_uncertainties = []
        scale_attention_weights = []
        
        # Process each scale
        for i, (patch_embedding, head, uncertainty_head) in enumerate(
            zip(self.patch_embeddings, self.heads, self.uncertainty_heads)
        ):
            # Patch embedding
            patch_embeds, num_patches = patch_embedding(x)
            
            # Transform through layers
            transformed = patch_embeds
            scale_attns = []
            for layer in self.transformer_layers:
                transformed, attn_weights = layer(transformed)
                scale_attns.append(attn_weights)
            
            # Generate predictions and uncertainties for this scale
            patch_preds = head(transformed)
            patch_uncs = uncertainty_head(transformed)
            
            # Global uncertainty context
            global_ctx = self.global_uncertainty(
                transformed.mean(dim=1)  # Average across patches
            ).unsqueeze(1).unsqueeze(2)
            
            patch_uncs = patch_uncs * global_ctx
            
            # Reconstruct from patches
            pred = self.reconstruct_from_patches(patch_preds, seq_len, self.strides[i], self.patch_lens[i])
            unc = self.reconstruct_from_patches(patch_uncs, seq_len, self.strides[i], self.patch_lens[i])
            
            scale_predictions.append(pred)
            scale_uncertainties.append(unc)
            scale_attention_weights.append(scale_attns)
        
        # Fuse multi-scale predictions
        stacked_preds = torch.stack(scale_predictions, dim=-1)  # (batch_size, seq_len, num_scales)
        stacked_uncs = torch.stack(scale_uncertainties, dim=-1)
        
        # Learn fusion weights
        fusion_weights = self.fusion(torch.ones_like(stacked_preds))  # (batch_size, seq_len, num_scales)
        
        # Weighted combination
        final_prediction = torch.sum(stacked_preds * fusion_weights, dim=-1)
        final_uncertainty = torch.sum(stacked_uncs * fusion_weights, dim=-1)
        
        return final_prediction, final_uncertainty, scale_attention_weights
    
    def reconstruct_from_patches(self, patches, target_len, stride, patch_len):
        """Reconstruct sequence from patches"""
        batch_size, num_patches, patch_size = patches.shape
        
        reconstruction = torch.zeros(batch_size, target_len, device=patches.device)
        counts = torch.zeros(batch_size, target_len, device=patches.device)
        
        for i in range(num_patches):
            start_idx = i * stride
            end_idx = start_idx + patch_len
            
            if end_idx <= target_len:
                reconstruction[:, start_idx:end_idx] += patches[:, i, :]
                counts[:, start_idx:end_idx] += 1
        
        reconstruction = reconstruction / (counts + 1e-8)
        return reconstruction

class PatchTSTTrainer:
    """PatchTST 학습 클래스"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 model_type='standard'):
        self.model = model.to(device)
        self.device = device
        self.model_type = model_type
        self.history = {'train_loss': [], 'val_loss': [], 'train_mse': [], 'val_mse': []}
    
    def patchtst_loss(self, pred, target, uncertainty, alpha=1.0, beta=0.1):
        """PatchTST-specific loss with patch consistency"""
        # Main reconstruction loss
        mse_loss = F.mse_loss(pred, target)
        
        # Uncertainty-aware loss
        precision = 1.0 / (uncertainty + 1e-8)
        uncertainty_loss = torch.mean(precision * (pred - target) ** 2)
        uncertainty_reg = torch.mean(torch.log(uncertainty + 1e-8))
        
        # Patch consistency loss (local smoothness within patches)
        patch_len = 16  # Default patch length
        num_patches = pred.shape[1] // patch_len
        
        patch_consistency_loss = 0
        for i in range(num_patches):
            start_idx = i * patch_len
            end_idx = start_idx + patch_len
            if end_idx <= pred.shape[1]:
                pred_patch = pred[:, start_idx:end_idx]
                target_patch = target[:, start_idx:end_idx]
                
                # Local smoothness within patch
                pred_diff = pred_patch[:, 1:] - pred_patch[:, :-1]
                target_diff = target_patch[:, 1:] - target_patch[:, :-1]
                patch_consistency_loss += F.mse_loss(pred_diff, target_diff)
        
        patch_consistency_loss /= max(num_patches, 1)
        
        # High-frequency preservation
        pred_fft = torch.fft.rfft(pred, dim=1)
        target_fft = torch.fft.rfft(target, dim=1)
        
        # Focus on preserving important frequency components
        freq_loss = F.mse_loss(pred_fft.real, target_fft.real) + \
                   F.mse_loss(pred_fft.imag, target_fft.imag)
        
        total_loss = uncertainty_loss + alpha * uncertainty_reg + \
                    beta * patch_consistency_loss + 0.1 * freq_loss
        
        return total_loss, mse_loss, uncertainty_reg, patch_consistency_loss, freq_loss
    
    def train_epoch(self, train_loader, optimizer, alpha=1.0, beta=0.1):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0
        total_mse = 0
        total_unc_reg = 0
        total_patch_loss = 0
        total_freq_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            
            optimizer.zero_grad()
            
            pred, uncertainty, _ = self.model(data)
            loss, mse_loss, unc_reg, patch_loss, freq_loss = self.patchtst_loss(
                pred, target, uncertainty, alpha, beta
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_unc_reg += unc_reg.item()
            total_patch_loss += patch_loss.item()
            total_freq_loss += freq_loss.item()
        
        return (total_loss / len(train_loader), 
                total_mse / len(train_loader),
                total_unc_reg / len(train_loader),
                total_patch_loss / len(train_loader),
                total_freq_loss / len(train_loader))
    
    def validate(self, val_loader, alpha=1.0, beta=0.1):
        """검증"""
        self.model.eval()
        total_loss = 0
        total_mse = 0
        total_unc_reg = 0
        total_patch_loss = 0
        total_freq_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                
                pred, uncertainty, _ = self.model(data)
                loss, mse_loss, unc_reg, patch_loss, freq_loss = self.patchtst_loss(
                    pred, target, uncertainty, alpha, beta
                )
                
                total_loss += loss.item()
                total_mse += mse_loss.item()
                total_unc_reg += unc_reg.item()
                total_patch_loss += patch_loss.item()
                total_freq_loss += freq_loss.item()
        
        return (total_loss / len(val_loader), 
                total_mse / len(val_loader),
                total_unc_reg / len(val_loader),
                total_patch_loss / len(val_loader),
                total_freq_loss / len(val_loader))
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.0001, alpha=1.0, beta=0.1,
              patience=15, save_path='best_patchtst.pth'):
        """전체 학습 프로세스"""
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.1)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting PatchTST ({self.model_type}) training...")
        
        for epoch in range(epochs):
            # Training
            train_loss, train_mse, train_unc, train_patch, train_freq = self.train_epoch(
                train_loader, optimizer, alpha, beta
            )
            
            # Validation
            val_loss, val_mse, val_unc, val_patch, val_freq = self.validate(
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
            print(f'  Train Loss: {train_loss:.6f} (MSE: {train_mse:.6f}, Unc: {train_unc:.6f}, Patch: {train_patch:.6f}, Freq: {train_freq:.6f})')
            print(f'  Val Loss: {val_loss:.6f} (MSE: {val_mse:.6f}, Unc: {val_unc:.6f}, Patch: {val_patch:.6f}, Freq: {val_freq:.6f})')
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
                mc_attentions = []
                
                for _ in range(n_samples):
                    pred, unc, attn = self.model(data)
                    mc_predictions.append(pred.cpu().numpy())
                    mc_uncertainties.append(unc.cpu().numpy())
                    if attn is not None:
                        mc_attentions.append([layer_attn.cpu().numpy() for layer_attn in attn])
                
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
                    attention_weights.append(mc_attentions)
        
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

def run_patchtst_experiment(data_dict, model_type='standard', batch_size=32, 
                           epochs=100, learning_rate=0.0001):
    """PatchTST 실험 실행 함수"""
    
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
    
    if model_type == 'multiscale':
        model = MultiscalePatchTST(
            seq_len=seq_len,
            patch_lens=[8, 16, 32],
            strides=[4, 8, 16],
            d_model=512,
            n_heads=8,
            n_layers=3,
            d_ff=2048,
            dropout=0.1
        )
    else:
        model = PatchTST(
            seq_len=seq_len,
            patch_len=16,
            stride=8,
            d_model=512,
            n_heads=8,
            n_layers=3,
            d_ff=2048,
            dropout=0.1,
            activation='gelu'
        )
    
    trainer = PatchTSTTrainer(model, device, model_type=model_type)
    
    # 학습
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=learning_rate,
        alpha=1.0,  # Uncertainty regularization weight
        beta=0.1,   # Patch consistency weight
        patience=15,
        save_path=f'best_patchtst_{model_type}.pth'
    )
    
    # 최고 모델 로드
    checkpoint = torch.load(f'best_patchtst_{model_type}.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 테스트 및 불확실성 추정
    print("\nRunning inference with uncertainty quantification...")
    results = trainer.predict_with_uncertainty(test_loader, n_samples=50)
    
    # 메트릭 계산
    predictions = results['predictions']
    targets = results['targets']
    total_uncertainty = results['total_uncertainty']
    
    metrics = calculate_metrics(targets, predictions, total_uncertainty)
    
    print(f"\n=== PatchTST ({model_type}) Results ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")
    
    # 결과 시각화
    plot_results(
        targets, predictions, total_uncertainty, 
        title=f"PatchTST ({model_type}) Denoising Results", sample_idx=0
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

# Standard PatchTST 실험
patchtst_results = run_patchtst_experiment(
    data_dict=data_dict,
    model_type='standard',
    batch_size=32,
    epochs=100,
    learning_rate=0.0001
)

# Multi-scale PatchTST 실험
multiscale_patchtst_results = run_patchtst_experiment(
    data_dict=data_dict,
    model_type='multiscale',
    batch_size=32,
    epochs=100,
    learning_rate=0.0001
)
"""
