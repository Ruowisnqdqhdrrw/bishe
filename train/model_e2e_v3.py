"""model_e2e_v3.py

优化版端到端公式识别模型

主要改进：
1. SE注意力增强的残差块
2. 真正的覆盖注意力机制（Coverage Attention）防止重复/遗漏
3. 更深的Encoder（可配置层数）
4. 多层LSTM Decoder with 残差连接 (LSTMDecoder)
5. Transformer Decoder with cross attention (TransformerDecoderWrapper)
6. 位置编码增强
7. DropPath正则化
8. KV Cache加速推理：自回归解码从O(T²)降至O(T)
"""

from __future__ import annotations

import math
import random
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """DropPath正则化：随机丢弃整个残差分支"""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation注意力模块"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlockV3(nn.Module):
    """增强版残差块：带SE注意力和DropPath"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 dropout: float = 0.1, drop_path: float = 0.0, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        # SE注意力
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.drop_path(out)
        out += self.shortcut(x)
        out = F.gelu(out)
        return out


class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EncoderCNNV3(nn.Module):
    """增强版CNN Encoder

    优化说明：
    - 移除了Encoder中的位置编码，避免与Decoder位置编码冲突
    - CNN本身的空间结构 + BiGRU的序列建模已提供足够的位置信息
    - Decoder（LSTM的隐状态传递 / Transformer的PE）负责解码端的位置感知
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 512,
        dropout: float = 0.2,
        drop_path: float = 0.1,
        use_se: bool = True,
    ) -> None:
        super().__init__()

        # 初始卷积层（更强的特征提取）
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        # 计算每层的drop_path率（逐层递增）
        num_blocks = 8
        dpr = [x.item() for x in torch.linspace(0, drop_path, num_blocks)]

        # 残差块组（更深的网络）
        self.layer1 = self._make_layer(64, 128, num_blocks=2, stride=2,
                                       dropout=dropout, drop_path=dpr[0:2], use_se=use_se)
        self.layer2 = self._make_layer(128, 256, num_blocks=2, stride=2,
                                       dropout=dropout, drop_path=dpr[2:4], use_se=use_se)
        self.layer3 = self._make_layer(256, 384, num_blocks=2, stride=2,
                                       dropout=dropout, drop_path=dpr[4:6], use_se=use_se)
        self.layer4 = self._make_layer(384, out_channels, num_blocks=2, stride=(2, 1),
                                       dropout=dropout, drop_path=dpr[6:8], use_se=use_se)

        # 最终池化：高度压缩到1
        self.final_pool = nn.AdaptiveAvgPool2d((1, None))

        # 双向GRU（3层，更强的序列建模）
        self.bigru = nn.GRU(
            input_size=out_channels,
            hidden_size=out_channels // 2,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.layer_norm = nn.LayerNorm(out_channels)
        self.out_dropout = nn.Dropout(p=dropout)
    
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, 
                    stride, dropout: float, drop_path: List[float], use_se: bool):
        if isinstance(stride, int):
            stride = (stride, stride)
        
        layers = [ResidualBlockV3(in_channels, out_channels, stride=stride[0], 
                                  dropout=dropout, drop_path=drop_path[0], use_se=use_se)]
        for i in range(1, num_blocks):
            dp = drop_path[i] if i < len(drop_path) else drop_path[-1]
            layers.append(ResidualBlockV3(out_channels, out_channels, stride=1, 
                                          dropout=dropout, drop_path=dp, use_se=use_se))
        
        if stride[1] != stride[0]:
            layers.append(nn.MaxPool2d(kernel_size=(stride[1], 1), stride=(stride[1], 1)))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, H, W]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.final_pool(x)  # [B, C, 1, L]
        x = x.squeeze(2)  # [B, C, L]
        x = x.permute(0, 2, 1)  # [B, L, C]

        # 双向GRU（不再添加位置编码，避免与Decoder的PE冲突）
        x, _ = self.bigru(x)  # [B, L, C]
        x = self.layer_norm(x)
        x = self.out_dropout(x)

        return x


class CoverageAttention(nn.Module):
    """覆盖注意力机制：防止重复关注和遗漏"""
    
    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int = 256, 
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.attention_dim = attention_dim
        
        assert self.head_dim * num_heads == attention_dim, "attention_dim must be divisible by num_heads"
        
        # 投影层
        self.query = nn.Linear(decoder_dim, attention_dim)
        self.key = nn.Linear(encoder_dim, attention_dim)
        self.value = nn.Linear(encoder_dim, attention_dim)
        
        # 覆盖特征：将累积注意力转换为特征
        self.coverage_fc = nn.Linear(1, attention_dim)
        
        # 输出投影
        self.out = nn.Linear(attention_dim, decoder_dim)
        
        self.dropout = nn.Dropout(p=dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, encoder_out: torch.Tensor, hidden: torch.Tensor,
                coverage: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_out: [B, L, D_enc]
            hidden: [B, D_dec]
            coverage: [B, L] 累积注意力权重
        Returns:
            context: [B, D_dec]
            alpha: [B, L] 当前注意力权重
        """
        batch_size = hidden.size(0)
        seq_len = encoder_out.size(1)
        
        # Query, Key, Value投影
        q = self.query(hidden).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(encoder_out).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(encoder_out).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, H, 1, L]
        
        # 添加覆盖特征（防止重复关注）
        if coverage is not None:
            # 将覆盖向量转换为特征并添加到注意力分数
            coverage_feat = self.coverage_fc(coverage.unsqueeze(-1))  # [B, L, attention_dim]
            coverage_feat = coverage_feat.view(batch_size, seq_len, self.num_heads, self.head_dim)
            coverage_feat = coverage_feat.permute(0, 2, 1, 3)  # [B, H, L, head_dim]
            
            # 覆盖惩罚：已经关注过的位置降低分数
            coverage_penalty = torch.sum(coverage_feat * k, dim=-1, keepdim=True)  # [B, H, L, 1]
            coverage_penalty = coverage_penalty.transpose(-2, -1)  # [B, H, 1, L]
            scores = scores - 0.1 * torch.tanh(coverage_penalty)
        
        # Softmax得到注意力权重
        alpha = F.softmax(scores, dim=-1)
        alpha = self.dropout(alpha)
        
        # 加权求和
        context = torch.matmul(alpha, v)  # [B, H, 1, head_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1)  # [B, attention_dim]
        context = self.out(context)  # [B, D_dec]
        
        # 返回平均注意力权重用于覆盖累积
        alpha_mean = alpha.squeeze(2).mean(dim=1)  # [B, L]
        
        return context, alpha_mean


class LSTMDecoder(nn.Module):
    """LSTM Attention Decoder（原 AttentionDecoderV3）"""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        encoder_dim: int,
        decoder_dim: int,
        attention_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.num_layers = num_layers
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(p=dropout)
        self.embed_norm = nn.LayerNorm(embed_dim)
        
        # 覆盖注意力
        self.attention = CoverageAttention(
            encoder_dim, decoder_dim, attention_dim, num_heads, dropout
        )
        
        # 多层LSTM
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        # 第一层：输入是embedding + context
        self.lstm_layers.append(nn.LSTMCell(embed_dim + decoder_dim, decoder_dim))
        self.layer_norms.append(nn.LayerNorm(decoder_dim))
        
        # 后续层：输入是上一层的输出
        for _ in range(1, num_layers):
            self.lstm_layers.append(nn.LSTMCell(decoder_dim, decoder_dim))
            self.layer_norms.append(nn.LayerNorm(decoder_dim))
        
        # 输出层（更深的MLP）
        self.fc = nn.Sequential(
            nn.Linear(decoder_dim + decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(decoder_dim, decoder_dim // 2),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(decoder_dim // 2, vocab_size),
        )
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        hiddens: List[Tuple[torch.Tensor, torch.Tensor]],
        encoder_out: torch.Tensor,
        coverage: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """
        Args:
            x: [B] 上一轮的token
            hiddens: List of (h, c) for each LSTM layer
            encoder_out: [B, L, D_enc]
            coverage: [B, L] 累积注意力
        Returns:
            output: [B, vocab_size]
            new_hiddens: List of (h, c)
            alpha: [B, L]
        """
        # Embedding
        embedded = self.embedding(x)  # [B, D_emb]
        embedded = self.embed_norm(embedded)
        embedded = self.embed_dropout(embedded)
        
        # 注意力（使用第一层的hidden state）
        context, alpha = self.attention(encoder_out, hiddens[0][0], coverage)
        
        # 多层LSTM
        new_hiddens = []
        lstm_input = torch.cat([embedded, context], dim=1)
        
        for i, (lstm, ln) in enumerate(zip(self.lstm_layers, self.layer_norms)):
            h, c = lstm(lstm_input, hiddens[i])
            h = ln(h)
            h = self.dropout(h)
            
            # 残差连接（从第二层开始）
            if i > 0:
                h = h + lstm_input
            
            new_hiddens.append((h, c))
            lstm_input = h
        
        # 输出：结合最后一层hidden和context
        output = self.fc(torch.cat([lstm_input, context], dim=1))
        
        return output, new_hiddens, alpha


class DecoderLayerWithCache(nn.Module):
    """Transformer decoder layer with KV cache support for O(T) incremental decoding.

    Drop-in replacement for nn.TransformerDecoderLayer that caches self-attention
    and cross-attention key/value tensors across decoding steps.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # Feed-forward
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard forward (no cache) — used during training."""
        # Pre-norm self-attention
        x = self.norm1(tgt)
        x, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(x)

        # Pre-norm cross-attention
        x = self.norm2(tgt)
        x, _ = self.cross_attn(x, memory, memory)
        tgt = tgt + self.dropout2(x)

        # Pre-norm feed-forward
        x = self.norm3(tgt)
        tgt = tgt + self.dropout3(self.ff(x))
        return tgt

    def forward_with_cache(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        sa_cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
        ca_cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Incremental forward: only processes the new token(s) using cached KV.

        Args:
            tgt: [B, 1, D] — embedding of the new token only
            memory: [B, L_enc, D]
            sa_cache: (key, value) each [B, prev_len, D] from previous steps, or None
            ca_cache: (key, value) each [B, L_enc, D] from first step, or None
        Returns:
            out: [B, 1, D]
            new_sa_cache: (key, value) each [B, prev_len+1, D]
            new_ca_cache: (key, value) each [B, L_enc, D]
        """
        # --- Self-attention with cache ---
        x = self.norm1(tgt)  # [B, 1, D]
        # Append current step to cached keys/values
        if sa_cache is not None:
            sa_k = torch.cat([sa_cache[0], x], dim=1)  # [B, prev+1, D]
            sa_v = torch.cat([sa_cache[1], x], dim=1)
        else:
            sa_k = x
            sa_v = x
        new_sa_cache = (sa_k, sa_v)
        # Query is only the new token; keys/values are the full history
        x, _ = self.self_attn(x, sa_k, sa_v)  # no causal mask needed: query is last position
        tgt = tgt + self.dropout1(x)

        # --- Cross-attention with cache ---
        x = self.norm2(tgt)
        if ca_cache is not None:
            ca_k, ca_v = ca_cache
        else:
            # First step: project memory through cross-attn in_proj for K, V
            ca_k = memory
            ca_v = memory
        new_ca_cache = (ca_k, ca_v)
        x, _ = self.cross_attn(x, ca_k, ca_v)
        tgt = tgt + self.dropout2(x)

        # --- Feed-forward ---
        x = self.norm3(tgt)
        tgt = tgt + self.dropout3(self.ff(x))

        return tgt, new_sa_cache, new_ca_cache


class TransformerDecoderWrapper(nn.Module):
    """Transformer Decoder with cross attention to CNN encoder output.

    Uses DecoderLayerWithCache for O(T) incremental inference decoding.
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_dim: int = 512,
        hidden_dim: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.2,
        max_seq_len: int = 256,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers

        # Token embedding + positional encoding
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=max_seq_len, dropout=dropout)
        self.embed_scale = math.sqrt(hidden_dim)

        # Project encoder output to hidden_dim if dimensions differ
        self.enc_proj = nn.Linear(encoder_dim, hidden_dim) if encoder_dim != hidden_dim else nn.Identity()

        # Custom decoder layers with KV cache support
        self.layers = nn.ModuleList([
            DecoderLayerWithCache(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Output projection
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        # Causal mask cache
        self._causal_mask: Optional[torch.Tensor] = None

    def _get_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        if self._causal_mask is None or self._causal_mask.size(0) < sz:
            mask = nn.Transformer.generate_square_subsequent_mask(sz, device=device)
            self._causal_mask = mask.bool()
        return self._causal_mask[:sz, :sz].to(device)

    def forward_train(
        self,
        encoder_out: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forcing forward pass.

        Args:
            encoder_out: [B, L_enc, encoder_dim]
            targets: [B, T] token ids (including <bos> prefix)
        Returns:
            logits: [B, T, vocab_size]
        """
        # Shift: input is targets[:, :-1], labels are targets[:, 1:]
        tgt_input = targets[:, :-1]  # [B, T-1]
        tgt_len = tgt_input.size(1)

        # Embed target tokens
        tgt_emb = self.embedding(tgt_input) * self.embed_scale  # [B, T-1, D]
        tgt_emb = self.pos_encoding(tgt_emb)

        # Project encoder memory
        memory = self.enc_proj(encoder_out)  # [B, L_enc, D]

        # Causal mask for target
        causal_mask = self._get_causal_mask(tgt_len, tgt_input.device)

        # Padding mask for target (True = ignore), cast to float to match causal_mask dtype
        tgt_pad_mask = (tgt_input == 0)  # [B, T-1]

        # Decode through custom layers (full-sequence, no cache)
        out = tgt_emb
        for layer in self.layers:
            out = layer(out, memory, tgt_mask=causal_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.final_norm(out)
        logits = self.fc_out(out)  # [B, T-1, vocab_size]
        return logits

    @torch.no_grad()
    def greedy_decode(
        self,
        encoder_out: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = 256,
    ) -> torch.Tensor:
        """Greedy autoregressive decoding with KV cache — O(T·D) instead of O(T²·D)."""
        device = encoder_out.device
        batch_size = encoder_out.size(0)
        memory = self.enc_proj(encoder_out)

        # Start with <bos>
        generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Initialize KV caches: one (sa_cache, ca_cache) per layer
        layer_caches: List[Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]],
                                  Optional[Tuple[torch.Tensor, torch.Tensor]]]] = [
            (None, None) for _ in range(self.num_layers)
        ]

        # Position counter for positional encoding
        pos = 0

        for _ in range(max_len - 1):
            # Only embed the last token
            last_token = generated[:, -1:]  # [B, 1]
            tok_emb = self.embedding(last_token) * self.embed_scale  # [B, 1, D]
            # Add positional encoding for current position only
            tok_emb = tok_emb + self.pos_encoding.pe[:, pos:pos + 1, :]
            tok_emb = self.pos_encoding.dropout(tok_emb)
            pos += 1

            # Pass through each layer with cache
            out = tok_emb
            new_caches = []
            for i, layer in enumerate(self.layers):
                sa_cache, ca_cache = layer_caches[i]
                out, new_sa, new_ca = layer.forward_with_cache(out, memory, sa_cache, ca_cache)
                new_caches.append((new_sa, new_ca))
            layer_caches = new_caches

            out = self.final_norm(out)  # [B, 1, D]
            next_logits = self.fc_out(out[:, 0, :])  # [B, V]
            next_id = next_logits.argmax(dim=-1, keepdim=True)  # [B, 1]

            # Force pad for already-finished sequences
            next_id = next_id.masked_fill(finished.unsqueeze(1), 0)
            generated = torch.cat([generated, next_id], dim=1)

            finished = finished | (next_id.squeeze(1) == eos_id)
            if finished.all():
                break

        # Pad to max_len
        if generated.size(1) < max_len:
            pad = torch.zeros(batch_size, max_len - generated.size(1), dtype=torch.long, device=device)
            generated = torch.cat([generated, pad], dim=1)
        else:
            generated = generated[:, :max_len]

        return generated

    @torch.no_grad()
    def beam_search_decode(
        self,
        encoder_out: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = 256,
        beam_width: int = 10,
        length_penalty: float = 0.6,
    ) -> torch.Tensor:
        """批量化beam search解码 with KV cache — O(T·D) instead of O(T²·D).

        相比旧版逐样本循环，将所有样本的所有beam合并为一个大batch
        一次性通过Transformer decoder，充分利用GPU并行能力。
        KV cache避免每步重算全部历史的self-attention和cross-attention。
        """
        device = encoder_out.device
        batch_size = encoder_out.size(0)
        bw = beam_width
        total = batch_size * bw
        memory_all = self.enc_proj(encoder_out)  # [B, L_enc, D]

        # 将memory扩展为 [B*beam, L_enc, D]
        memory_exp = memory_all.unsqueeze(1).expand(
            -1, bw, -1, -1
        ).reshape(total, memory_all.size(1), memory_all.size(2))

        # 初始化序列和分数
        alive_seqs = torch.full((total, 1), bos_id, dtype=torch.long, device=device)
        alive_scores = torch.zeros(total, device=device)
        # 初始时只有每个样本的第一个beam有效
        for k in range(1, bw):
            alive_scores[k::bw] = -1e9

        finished = torch.zeros(total, dtype=torch.bool, device=device)
        best_finished_scores = torch.full((batch_size,), -1e9, device=device)
        best_finished_seqs = [[bos_id, eos_id]] * batch_size

        # KV caches: list of (sa_cache, ca_cache) per layer
        # Each cache tensor has shape [B*beam, seq_len, D]
        layer_caches: List[Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]],
                                  Optional[Tuple[torch.Tensor, torch.Tensor]]]] = [
            (None, None) for _ in range(self.num_layers)
        ]
        pos = 0  # position counter for PE

        for step in range(max_len - 1):
            if finished.all():
                break

            # Only embed the last token
            last_token = alive_seqs[:, -1:]  # [B*beam, 1]
            tok_emb = self.embedding(last_token) * self.embed_scale
            tok_emb = tok_emb + self.pos_encoding.pe[:, pos:pos + 1, :]
            tok_emb = self.pos_encoding.dropout(tok_emb)
            pos += 1

            # Incremental forward through each layer
            out = tok_emb
            new_caches = []
            for i, layer in enumerate(self.layers):
                sa_cache, ca_cache = layer_caches[i]
                out, new_sa, new_ca = layer.forward_with_cache(out, memory_exp, sa_cache, ca_cache)
                new_caches.append((new_sa, new_ca))
            layer_caches = new_caches

            out = self.final_norm(out)
            next_logits = self.fc_out(out[:, 0, :])  # [B*beam, V]
            log_probs = F.log_softmax(next_logits, dim=-1)

            vocab_size = log_probs.size(-1)

            # 已完成的beam不再扩展
            if finished.any():
                log_probs[finished] = -1e9
                log_probs[finished, 0] = 0.0

            candidate_scores = alive_scores.unsqueeze(1) + log_probs
            candidate_scores = candidate_scores.view(batch_size, bw * vocab_size)

            topk_scores, topk_indices = candidate_scores.topk(bw, dim=-1)

            beam_indices = topk_indices // vocab_size
            token_indices = topk_indices % vocab_size

            batch_offsets = torch.arange(batch_size, device=device).unsqueeze(1) * bw
            global_beam_idx = (batch_offsets + beam_indices).view(-1)  # [B*beam]

            # 更新序列
            new_seqs = alive_seqs[global_beam_idx]
            new_tokens = token_indices.view(-1, 1)
            alive_seqs = torch.cat([new_seqs, new_tokens], dim=1)

            alive_scores = topk_scores.view(-1)

            # Reorder KV caches to match the selected beams
            reordered_caches = []
            for sa_cache, ca_cache in layer_caches:
                new_sa = (sa_cache[0][global_beam_idx], sa_cache[1][global_beam_idx])
                new_ca = (ca_cache[0][global_beam_idx], ca_cache[1][global_beam_idx])
                reordered_caches.append((new_sa, new_ca))
            layer_caches = reordered_caches

            # 更新finished状态
            new_finished = (new_tokens.squeeze(1) == eos_id)
            finished = finished[global_beam_idx] | new_finished

            # 收集已完成的序列
            for b_idx in range(batch_size):
                start = b_idx * bw
                for k in range(bw):
                    idx = start + k
                    if new_finished[idx]:
                        sl = alive_seqs.size(1)
                        lp = ((5.0 + sl) / 6.0) ** length_penalty
                        norm_score = alive_scores[idx].item() / lp
                        if norm_score > best_finished_scores[b_idx].item():
                            best_finished_scores[b_idx] = norm_score
                            best_finished_seqs[b_idx] = alive_seqs[idx].tolist()

            # 提前终止检查
            if (best_finished_scores > -1e8).all():
                alive_best = alive_scores.view(batch_size, bw).max(dim=1).values
                cur_len_val = alive_seqs.size(1)
                lp_cur = ((5.0 + cur_len_val) / 6.0) ** length_penalty
                alive_best_norm = alive_best / lp_cur
                if (alive_best_norm < best_finished_scores).all():
                    break

        # 对于没有完成序列的样本，取分数最高的活跃beam
        result = []
        for b_idx in range(batch_size):
            if best_finished_scores[b_idx].item() > -1e8:
                seq = best_finished_seqs[b_idx]
            else:
                start = b_idx * bw
                seq = alive_seqs[start].tolist()

            if len(seq) < max_len:
                seq = seq + [0] * (max_len - len(seq))
            else:
                seq = seq[:max_len]
            result.append(seq)

        return torch.tensor(result, dtype=torch.long, device=device)


# Keep backward-compatible alias
AttentionDecoderV3 = LSTMDecoder


class E2EModelV3(nn.Module):
    """优化版端到端模型（支持 LSTM / Transformer decoder）"""
    
    def __init__(self, vocab_size: int, cfg: dict) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = cfg["encoder_dim"]
        self.decoder_type = cfg.get("decoder_type", "lstm")

        self.encoder = EncoderCNNV3(
            in_channels=1,
            out_channels=cfg["encoder_dim"],
            dropout=cfg.get("encoder_dropout", 0.2),
            drop_path=cfg.get("drop_path", 0.1),
            use_se=cfg.get("use_se", True),
        )

        if self.decoder_type == "transformer":
            # ---------- Transformer decoder ----------
            self.decoder_dim = cfg.get("transformer_hidden_dim", 512)
            self.decoder = TransformerDecoderWrapper(
                vocab_size=vocab_size,
                encoder_dim=cfg["encoder_dim"],
                hidden_dim=cfg.get("transformer_hidden_dim", 512),
                nhead=cfg.get("transformer_nhead", 8),
                num_layers=cfg.get("transformer_num_layers", 4),
                dropout=cfg.get("transformer_dropout", 0.2),
                max_seq_len=cfg.get("max_seq_len", 256),
            )
            # No init_h/init_c or coverage for transformer
            self.init_h = None
            self.init_c = None
            self.coverage_weight = 0.0
        else:
            # ---------- LSTM decoder (original) ----------
            self.decoder_dim = cfg["decoder_dim"]
            self.num_decoder_layers = cfg.get("decoder_layers", 3)
            self.decoder = LSTMDecoder(
                vocab_size=vocab_size,
                embed_dim=cfg["embed_dim"],
                encoder_dim=cfg["encoder_dim"],
                decoder_dim=cfg["decoder_dim"],
                attention_dim=cfg.get("attention_dim", 256),
                num_heads=cfg.get("num_heads", 8),
                num_layers=self.num_decoder_layers,
                dropout=cfg["dropout"],
            )
            self.init_h = nn.ModuleList([
                nn.Linear(cfg["encoder_dim"], cfg["decoder_dim"])
                for _ in range(self.num_decoder_layers)
            ])
            self.init_c = nn.ModuleList([
                nn.Linear(cfg["encoder_dim"], cfg["decoder_dim"])
                for _ in range(self.num_decoder_layers)
            ])
            self.coverage_weight = cfg.get("coverage_weight", 1.0)
    
    def _init_hidden(self, encoder_out: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """从encoder输出初始化decoder hidden states（仅 LSTM decoder 使用）"""
        if self.init_h is None:
            return []
        mean_enc = encoder_out.mean(dim=1)
        hiddens = []
        for init_h, init_c in zip(self.init_h, self.init_c):
            h = torch.tanh(init_h(mean_enc))
            c = torch.tanh(init_c(mean_enc))
            hiddens.append((h, c))
        return hiddens
    
    def forward(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: [B, 1, H, W]
            targets: [B, T_max]
            teacher_forcing_ratio: teacher forcing概率 (ignored for transformer)
        Returns:
            outputs: [B, T_max, vocab_size]
            coverage_loss: 覆盖损失 (0 for transformer)
        """
        batch_size = images.size(0)
        max_len = targets.size(1)
        device = images.device

        encoder_out = self.encoder(images)

        if self.decoder_type == "transformer":
            # Transformer Scheduled Sampling:
            # tf_ratio=1.0 时使用并行teacher forcing（高效）
            # tf_ratio<1.0 时混入模型自身预测，逐步降低对GT的依赖
            if teacher_forcing_ratio >= 1.0:
                # 纯teacher forcing：并行前向，训练效率最高
                logits = self.decoder.forward_train(encoder_out, targets)  # [B, T-1, V]
                outputs = torch.zeros(batch_size, max_len, self.vocab_size, device=device)
                outputs[:, 1:logits.size(1)+1, :] = logits
            else:
                # Scheduled Sampling：按概率替换部分输入为模型预测
                # 先做一次完整的teacher forcing前向获取logits
                with torch.no_grad():
                    tf_logits = self.decoder.forward_train(encoder_out, targets)
                # 构造混合输入：随机将部分位置替换为模型预测的token
                mixed_input = targets.clone()
                pred_ids = tf_logits.argmax(dim=-1)  # [B, T-1]
                # 生成mask：True表示使用模型预测而非GT
                sample_mask = torch.rand(pred_ids.shape, device=device) > teacher_forcing_ratio
                # targets[:, 1:-1] 对应 pred_ids[:, :-1]（错位一个时间步）
                t_len = min(pred_ids.size(1), mixed_input.size(1) - 1)
                mixed_input[:, 1:t_len+1] = torch.where(
                    sample_mask[:, :t_len], pred_ids[:, :t_len], targets[:, 1:t_len+1]
                )
                # 用混合输入重新前向
                logits = self.decoder.forward_train(encoder_out, mixed_input)
                outputs = torch.zeros(batch_size, max_len, self.vocab_size, device=device)
                outputs[:, 1:logits.size(1)+1, :] = logits
            coverage_loss = torch.zeros(1, device=device)
        else:
            # LSTM: autoregressive with teacher forcing
            seq_len = encoder_out.size(1)
            hiddens = self._init_hidden(encoder_out)
            outputs = torch.zeros(batch_size, max_len, self.vocab_size, device=device)
            coverage = torch.zeros(batch_size, seq_len, device=device)
            coverage_loss = torch.zeros(1, device=device)

            decoder_input = targets[:, 0]  # <bos>

            for t in range(1, max_len):
                output, hiddens, alpha = self.decoder(
                    decoder_input, hiddens, encoder_out, coverage
                )
                outputs[:, t, :] = output

                coverage_loss = coverage_loss + torch.sum(torch.min(alpha, coverage))
                coverage = coverage + alpha

                use_teacher_forcing = random.random() < teacher_forcing_ratio
                if use_teacher_forcing:
                    decoder_input = targets[:, t]
                else:
                    decoder_input = output.argmax(1)

            coverage_loss = coverage_loss / (batch_size * max_len)

        return outputs, coverage_loss * self.coverage_weight
    
    @torch.no_grad()
    def greedy_decode(
        self,
        images: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = 256,
    ) -> torch.Tensor:
        """Greedy解码"""
        device = images.device
        batch_size = images.size(0)

        encoder_out = self.encoder(images)

        if self.decoder_type == "transformer":
            return self.decoder.greedy_decode(encoder_out, bos_id, eos_id, max_len)

        seq_len = encoder_out.size(1)
        hiddens = self._init_hidden(encoder_out)
        coverage = torch.zeros(batch_size, seq_len, device=device)

        preds = torch.full((batch_size, max_len), bos_id, dtype=torch.long, device=device)
        decoder_input = preds[:, 0]

        for t in range(1, max_len):
            output, hiddens, alpha = self.decoder(
                decoder_input, hiddens, encoder_out, coverage
            )
            coverage = coverage + alpha

            next_id = output.argmax(dim=1)
            preds[:, t] = next_id
            decoder_input = next_id

            if (next_id == eos_id).all():
                break

        return preds
    
    @torch.no_grad()
    def beam_search_decode(
        self,
        images: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = 256,
        beam_width: int = 10,
        length_penalty: float = 0.6,
        coverage_penalty: float = 0.0,
        repetition_penalty: float = 0.0,
    ) -> torch.Tensor:
        """Beam Search 解码

        score 归一化方式：score / (length ** length_penalty)
        coverage_penalty 和 repetition_penalty 默认关闭。
        """
        device = images.device
        batch_size = images.size(0)

        if self.decoder_type == "transformer":
            encoder_out = self.encoder(images)
            return self.decoder.beam_search_decode(
                encoder_out, bos_id, eos_id, max_len, beam_width, length_penalty,
            )

        all_preds = []

        for b in range(batch_size):
            img = images[b:b+1]
            encoder_out = self.encoder(img)
            seq_len = encoder_out.size(1)
            hiddens = self._init_hidden(encoder_out)
            coverage = torch.zeros(1, seq_len, device=device)

            # beam: (score, sequence, hiddens, coverage)
            # score = 累积 log-prob（未归一化）
            beams = [(0.0, [bos_id], hiddens, coverage)]
            completed = []  # (normalized_score, sequence)

            def _normalize_score(raw_score: float, seq_len: int, cov=None) -> float:
                """Google NMT 风格的长度归一化"""
                lp = ((5.0 + seq_len) / 6.0) ** length_penalty
                s = raw_score / lp
                if coverage_penalty > 0 and cov is not None:
                    # 标准 coverage penalty: 惩罚 under-attended 位置
                    # sum of min(cov, 1.0) — 完美覆盖时等于 encoder_len
                    cp = coverage_penalty * torch.sum(torch.clamp(cov, max=1.0)).item()
                    s += cp
                return s

            for t in range(1, max_len):
                all_candidates = []

                for score, seq, h_list, cov in beams:
                    if seq[-1] == eos_id:
                        ns = _normalize_score(score, len(seq), cov)
                        completed.append((ns, seq))
                        continue

                    decoder_input = torch.tensor([seq[-1]], dtype=torch.long, device=device)
                    output, new_h, alpha = self.decoder(
                        decoder_input, h_list, encoder_out, cov
                    )
                    new_cov = cov + alpha

                    log_probs = F.log_softmax(output, dim=-1)

                    # 可选：轻度重复惩罚
                    if repetition_penalty > 0 and len(seq) >= 3:
                        last = seq[-1]
                        if seq[-2] == last and seq[-3] == last:
                            if last != bos_id and last != eos_id:
                                log_probs[0, last] -= repetition_penalty

                    topk_probs, topk_ids = log_probs.topk(beam_width, dim=-1)

                    for i in range(beam_width):
                        token_id = topk_ids[0, i].item()
                        token_score = topk_probs[0, i].item()
                        new_score = score + token_score
                        new_seq = seq + [token_id]
                        all_candidates.append((new_score, new_seq, new_h, new_cov))

                if not all_candidates:
                    break

                # 用归一化分数排序，避免偏向短序列
                all_candidates.sort(
                    key=lambda x: _normalize_score(x[0], len(x[1]), x[3]),
                    reverse=True,
                )
                beams = all_candidates[:beam_width]

                # 所有 beam 都已结束
                if all(seq[-1] == eos_id for _, seq, _, _ in beams):
                    for sc, sq, _, cv in beams:
                        ns = _normalize_score(sc, len(sq), cv)
                        completed.append((ns, sq))
                    break

                # 足够多的完成序列
                if len(completed) >= beam_width:
                    break

            # 未完成的 beam 也加入候选
            for score, seq, _, cov in beams:
                if seq[-1] != eos_id:
                    ns = _normalize_score(score, len(seq), cov)
                    completed.append((ns, seq))

            if completed:
                completed.sort(key=lambda x: x[0], reverse=True)
                best_seq = completed[0][1]
            else:
                best_seq = [bos_id, eos_id]

            # Padding
            if len(best_seq) < max_len:
                best_seq = best_seq + [0] * (max_len - len(best_seq))
            else:
                best_seq = best_seq[:max_len]

            all_preds.append(best_seq)

        return torch.tensor(all_preds, dtype=torch.long, device=device)


def build_e2e_model_v3(vocab_size: int, cfg: dict) -> E2EModelV3:
    """构建优化版模型"""
    return E2EModelV3(vocab_size, cfg)
