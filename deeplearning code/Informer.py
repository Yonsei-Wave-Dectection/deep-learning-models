import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import math
from typing import Tuple, Dict

class ProbAttention(nn.Module):
    """ProbSparse Self-Attention mechanism"""
    
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
    
    def _prob_QK(self, Q, K, sample_k, n_top):
        """Compute probabilities for Q-K pairs"""
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        
        # Calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()
        
        # Find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        
        # Use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k
        
        return Q_K, M_top
    
    def _get_initial_context(self, V, L_Q):
        """Get initial context"""
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert(L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex
    
    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        """Update context with attention scores"""
        B, H, L_V, D = V.shape
        
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        
        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)
        
        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)
    
    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)
        
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)
        
        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        
        # Add scale factor
        scale = self.scale or 1./math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        
        # Get the context
        context = self._get_initial_context(values, L_Q)
        
        # Update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2, 1).contiguous(), attn

class ProbMask:
    """Probability mask for ProbSparse attention"""
    
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask

class AttentionLayer(nn.Module):
    """Attention layer wrapper"""
    
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, dropout=0.1):
        super(AttentionLayer, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        
        self.inner_attention = attention
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
        
        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        
        return self.dropout(self.out_projection(out)), attn

class ConvLayer(nn.Module):
    """Convolutional layer for distilling operation"""
    
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x

class EncoderLayer(nn.Module):
    """Informer encoder layer"""
    
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
    
    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm2(x + y), attn

class Encoder(nn.Module):
    """Informer encoder with distilling"""
    
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
    
    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x, attns

class DecoderLayer(nn.Module):
    """Informer decoder layer"""
    
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
    
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x)
        
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm3(x + y)

class Decoder(nn.Module):
    """Informer decoder"""
    
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
    
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x

class DataEmbedding(nn.Module):
    """Data embedding combining value, positional and temporal embeddings"""
    
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

class Informer(nn.Module):
    """Informer model for seismic signal denoising with uncertainty quantification"""
    
    def __init__(self, seq_len=1000, label_len=0, pred_len=0, 
                 factor=5, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=512, 
                 dropout=0.1, attn='prob', embed='fixed', activation='gelu', 
                 output_attention=False, distil=True):
        super(Informer, self).__init__()
        self.pred_len = pred_len
        self.label_len = label_len
        self.attn = attn
        self.output_attention = output_attention
        
        # Embedding
        self.enc_embedding = DataEmbedding(1, d_model, dropout)
        self.dec_embedding = DataEmbedding(1, d_model, dropout)
        
        # Attention mechanism
        Attn = ProbAttention if attn == 'prob' else FullAttention
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                 d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(d_model) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                 d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                 d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # Output layers
        self.projection = nn.Linear(d_model, 1, bias=True)
        
        # Uncertainty estimation
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
        
        # Global uncertainty context
        self.global_uncertainty = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        batch_size, seq_len = x.shape
        
        # Add feature dimension
        x = x.unsqueeze(-1)  # (batch_size, seq_len, 1)
        
        # For denoising, we use the noisy signal as both encoder and decoder input
        enc_out = self.enc_embedding(x)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        # Use encoder output as decoder input for denoising
        dec_out = self.dec_embedding(x)
        dec_out = self.decoder(dec_out, enc_out)
        
        # Generate predictions
        predictions = self.projection(dec_out).squeeze(-1)  # (batch_size, seq_len)
        
        # Generate local uncertainty
        local_uncertainty = self.uncertainty_head(dec_out).squeeze(-1)  # (batch_size, seq_len)
        
        # Generate global uncertainty context
        global_context = self.global_uncertainty(dec_out.transpose(1, 2)).squeeze(-1)  # (batch_size, 1)
        global_context = global_context.unsqueeze(1).expand(-1, seq_len)  # (batch_size, seq_len)
        
        # Combine uncertainties
        uncertainty = local_uncertainty * global_context
        
        if self.output_attention:
            return predictions, uncertainty, attns
        else:
            return predictions, uncertainty, None

class FullAttention(nn.Module):
    """Full attention mechanism as fallback"""
    
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
    
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./math.sqrt(E)
        
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class TriangularCausalMask:
    """Triangular causal mask"""
    
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
    
    @property
    def mask(self):
        return self._mask

class InformerTrainer:
    """Informer 학습 클래스"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {'train_loss': [], 'val_loss': [], 'train_mse': [], 'val_mse': []}
    
    def informer_loss(self, pred, target, uncertainty, alpha=1.0, beta=0.1):
        """Informer-specific loss with uncertainty and attention regularization"""
        # Main reconstruction loss
        mse_loss = F.mse_loss(pred, target)
        
        # Uncertainty-aware loss
        precision = 1.0 / (uncertainty + 1e-8)
        uncertainty_loss = torch.mean(precision * (pred - target) ** 2)
        uncertainty_reg = torch.mean(torch.log(uncertainty + 1e-8))
        
        # Temporal smoothness regularization
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        smoothness_loss = F.mse_loss(pred_diff, target_diff)
        
        # High-frequency preservation
        pred_fft = torch.fft.rfft(pred, dim=1)
        target_fft = torch.fft.rfft(target, dim=1)
        
        # Focus on mid to high frequencies for seismic signals
        mid_freq_start = pred_fft.shape[1] // 4
        high_freq_fft_loss = F.mse_loss(pred_fft[:, mid_freq_start:].real, target_fft[:, mid_freq_start:].real) + \
                            F.mse_loss(pred_fft[:, mid_freq_start:].imag, target_fft[:, mid_freq_start:].imag)
        
        total_loss = uncertainty_loss + alpha * uncertainty_reg + \
                    beta * smoothness_loss + 0.1 * high_freq_fft_loss
        
        return total_loss, mse_loss, uncertainty_reg, smoothness_loss, high_freq_fft_loss
    
    def train_epoch(self, train_loader, optimizer, alpha=1.0, beta=0.1):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0
        total_mse = 0
        total_unc_reg = 0
        total_smooth_loss = 0
        total_freq_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            
            optimizer.zero_grad()
            
            pred, uncertainty, _ = self.model(data)
            loss, mse_loss, unc_reg, smooth_loss, freq_loss = self.informer_loss(
                pred, target, uncertainty, alpha, beta
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_unc_reg += unc_reg.item()
            total_smooth_loss += smooth_loss.item()
            total_freq_loss += freq_loss.item()
        
        return (total_loss / len(train_loader), 
                total_mse / len(train_loader),
                total_unc_reg / len(train_loader),
                total_smooth_loss / len(train_loader),
                total_freq_loss / len(train_loader))
    
    def validate(self, val_loader, alpha=1.0, beta=0.1):
        """검증"""
        self.model.eval()
        total_loss = 0
        total_mse = 0
        total_unc_reg = 0
        total_smooth_loss = 0
        total_freq_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                
                pred, uncertainty, _ = self.model(data)
                loss, mse_loss, unc_reg, smooth_loss, freq_loss = self.informer_loss(
                    pred, target, uncertainty, alpha, beta
                )
                
                total_loss += loss.item()
                total_mse += mse_loss.item()
                total_unc_reg += unc_reg.item()
                total_smooth_loss += smooth_loss.item()
                total_freq_loss += freq_loss.item()
        
        return (total_loss / len(val_loader), 
                total_mse / len(val_loader),
                total_unc_reg / len(val_loader),
                total_smooth_loss / len(val_loader),
                total_freq_loss / len(val_loader))
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.0001, alpha=1.0, beta=0.1,
              patience=15, save_path='best_informer.pth'):
        """전체 학습 프로세스"""
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.1)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("Starting Informer training...")
        
        for epoch in range(epochs):
            # Training
            train_loss, train_mse, train_unc, train_smooth, train_freq = self.train_epoch(
                train_loader, optimizer, alpha, beta
            )
            
            # Validation
            val_loss, val_mse, val_unc, val_smooth, val_freq = self.validate(
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
            print(f'  Train Loss: {train_loss:.6f} (MSE: {train_mse:.6f}, Unc: {train_unc:.6f}, Smooth: {train_smooth:.6f}, Freq: {train_freq:.6f})')
            print(f'  Val Loss: {val_loss:.6f} (MSE: {val_mse:.6f}, Unc: {val_unc:.6f}, Smooth: {val_smooth:.6f}, Freq: {val_freq:.6f})')
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

def run_informer_experiment(data_dict, batch_size=32, epochs=100, learning_rate=0.0001):
    """Informer 실험 실행 함수"""
    
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
    
    model = Informer(
        seq_len=seq_len,
        label_len=0,
        pred_len=0,
        factor=5,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        dropout=0.1,
        attn='prob',
        embed='fixed',
        activation='gelu',
        output_attention=False,
        distil=True
    )
    
    trainer = InformerTrainer(model, device)
    
    # 학습
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=learning_rate,
        alpha=1.0,  # Uncertainty regularization weight
        beta=0.1,   # Smoothness loss weight
        patience=15,
        save_path='best_informer.pth'
    )
    
    # 최고 모델 로드
    checkpoint = torch.load('best_informer.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 테스트 및 불확실성 추정
    print("\nRunning inference with uncertainty quantification...")
    results = trainer.predict_with_uncertainty(test_loader, n_samples=50)
    
    # 메트릭 계산
    predictions = results['predictions']
    targets = results['targets']
    total_uncertainty = results['total_uncertainty']
    
    metrics = calculate_metrics(targets, predictions, total_uncertainty)
    
    print("\n=== Informer Results ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")
    
    # 결과 시각화
    plot_results(
        targets, predictions, total_uncertainty, 
        title="Informer Denoising Results", sample_idx=0
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

# Informer 실험
informer_results = run_informer_experiment(
    data_dict=data_dict,
    batch_size=16,  # Informer는 메모리를 많이 사용하므로 배치 크기를 줄임
    epochs=100,
    learning_rate=0.0001
)
"""
