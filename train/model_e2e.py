"""model_e2e.py

端到端公式识别模型（兼容旧版checkpoint）

功能概述：
- CNN Encoder（ResNet风格残差块）
- 双向GRU编码特征
- 多头注意力机制
- LSTM Decoder
"""

from __future__ import annotations

import math
import random
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """残差块（无SE注意力，兼容旧版本）"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class EncoderCNNv2(nn.Module):
    """CNN Encoder（兼容旧版本checkpoint）"""
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 512,
        dropout: float = 0.2,
        drop_path: float = 0.1,  # 保留参数但不使用
    ) -> None:
        super().__init__()
        
        # 初始卷积层（旧版本只有一个卷积+BN+ReLU）
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # 残差块组
        self.layer1 = self._make_layer(64, 128, num_blocks=2, stride=2, dropout=dropout)
        self.layer2 = self._make_layer(128, 256, num_blocks=2, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(256, 384, num_blocks=2, stride=2, dropout=dropout)
        self.layer4 = self._make_layer(384, out_channels, num_blocks=2, stride=(2, 1), dropout=dropout)
        
        # 最终池化：高度压缩到1
        self.final_pool = nn.AdaptiveAvgPool2d((1, None))
        
        # 双向GRU（旧版本是2层）
        self.bigru = nn.GRU(
            input_size=out_channels,
            hidden_size=out_channels // 2,
            num_layers=2,  # 旧版本是2层
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        
        self.layer_norm = nn.LayerNorm(out_channels)
        self.out_dropout = nn.Dropout(p=dropout)
    
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, 
                    stride, dropout: float):
        if isinstance(stride, int):
            stride = (stride, stride)
        
        layers = [ResidualBlock(in_channels, out_channels, stride=stride[0], dropout=dropout)]
        for i in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1, dropout=dropout))
        
        if stride[1] != stride[0]:
            layers.append(nn.MaxPool2d(kernel_size=(stride[1], 1), stride=(stride[1], 1)))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, H, W]
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.final_pool(x)  # [B, C, 1, L]
        x = x.squeeze(2)  # [B, C, L]
        x = x.permute(0, 2, 1)  # [B, L, C]
        
        # 双向GRU
        x, _ = self.bigru(x)  # [B, L, C]
        x = self.layer_norm(x)
        x = self.out_dropout(x)
        
        return x


class MultiHeadAttention(nn.Module):
    """多头注意力机制（无覆盖机制，兼容旧版本）"""
    
    def __init__(self, encoder_dim: int, decoder_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = decoder_dim // num_heads
        assert self.head_dim * num_heads == decoder_dim, "decoder_dim must be divisible by num_heads"
        
        self.query = nn.Linear(decoder_dim, decoder_dim)
        self.key = nn.Linear(encoder_dim, decoder_dim)
        self.value = nn.Linear(encoder_dim, decoder_dim)
        self.out = nn.Linear(decoder_dim, decoder_dim)
        
        self.dropout = nn.Dropout(p=dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, encoder_out: torch.Tensor, hidden: torch.Tensor,
                coverage: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # encoder_out: [B, L, D_enc]
        # hidden: [B, D_dec]
        # coverage: [B, L] 累积注意力（可选，这里忽略）
        
        batch_size = hidden.size(0)
        seq_len = encoder_out.size(1)
        
        q = self.query(hidden).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(encoder_out).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(encoder_out).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, H, 1, L]
        
        alpha = F.softmax(scores, dim=-1)
        alpha = self.dropout(alpha)
        
        context = torch.matmul(alpha, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1)
        context = self.out(context)
        
        return context, alpha.squeeze(2).mean(dim=1)  # [B, L]


class AttentionDecoderV2(nn.Module):
    """Attention Decoder（兼容旧版本checkpoint）"""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        encoder_dim: int,
        decoder_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(p=dropout)
        
        self.attention = MultiHeadAttention(encoder_dim, decoder_dim, num_heads, dropout)
        
        # 使用LSTM
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        self.lstm2 = nn.LSTMCell(decoder_dim, decoder_dim)
        
        self.layer_norm1 = nn.LayerNorm(decoder_dim)
        self.layer_norm2 = nn.LayerNorm(decoder_dim)
        
        # 输出层（旧版本结构：2层全连接）
        self.fc = nn.Sequential(
            nn.Linear(decoder_dim + encoder_dim, decoder_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(decoder_dim, vocab_size),
        )
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        hidden2: Tuple[torch.Tensor, torch.Tensor],
        encoder_out: torch.Tensor,
        coverage: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # x: [B] (上一轮的 token)
        # hidden: (h, c) for LSTM layer 1
        # hidden2: (h, c) for LSTM layer 2
        # encoder_out: [B, L, D_enc]
        # coverage: [B, L] 累积注意力
        
        # 注意力
        context, alpha = self.attention(encoder_out, hidden[0], coverage)  # [B, D_enc], [B, L]
        
        # Embedding
        embedded = self.embedding(x)  # [B, D_emb]
        embedded = self.embed_dropout(embedded)
        
        # LSTM Layer 1
        lstm_input = torch.cat([embedded, context], dim=1)
        h1, c1 = self.lstm(lstm_input, hidden)
        h1 = self.layer_norm1(h1)
        h1 = self.dropout(h1)
        
        # LSTM Layer 2 with residual
        h2, c2 = self.lstm2(h1, hidden2)
        h2 = self.layer_norm2(h2 + h1)  # 残差连接
        h2 = self.dropout(h2)
        
        # 输出：结合decoder状态和context
        output = self.fc(torch.cat([h2, context], dim=1))
        
        return output, (h1, c1), (h2, c2), alpha


class E2EModelV2(nn.Module):
    """端到端模型（兼容旧版本checkpoint）"""
    
    def __init__(self, vocab_size: int, cfg: dict) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = cfg["encoder_dim"]
        self.decoder_dim = cfg["decoder_dim"]
        
        self.encoder = EncoderCNNv2(
            in_channels=1,
            out_channels=cfg["encoder_dim"],
            dropout=cfg.get("encoder_dropout", 0.2),
            drop_path=cfg.get("drop_path", 0.1),
        )
        
        self.decoder = AttentionDecoderV2(
            vocab_size=vocab_size,
            embed_dim=cfg["embed_dim"],
            encoder_dim=cfg["encoder_dim"],
            decoder_dim=cfg["decoder_dim"],
            num_heads=cfg.get("num_heads", 8),
            num_layers=cfg.get("decoder_layers", 2),
            dropout=cfg["dropout"],
        )
        
        # 初始化hidden state的投影
        self.init_h = nn.Linear(cfg["encoder_dim"], cfg["decoder_dim"])
        self.init_c = nn.Linear(cfg["encoder_dim"], cfg["decoder_dim"])
        self.init_h2 = nn.Linear(cfg["encoder_dim"], cfg["decoder_dim"])
        self.init_c2 = nn.Linear(cfg["encoder_dim"], cfg["decoder_dim"])
        
        # 覆盖损失权重
        self.coverage_weight = cfg.get("coverage_weight", 0.1)
    
    def _init_hidden(self, encoder_out: torch.Tensor):
        """从encoder输出初始化decoder hidden state"""
        mean_enc = encoder_out.mean(dim=1)
        h1 = torch.tanh(self.init_h(mean_enc))
        c1 = torch.tanh(self.init_c(mean_enc))
        h2 = torch.tanh(self.init_h2(mean_enc))
        c2 = torch.tanh(self.init_c2(mean_enc))
        return (h1, c1), (h2, c2)
    
    def forward(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # images: [B, 1, H, W]
        # targets: [B, T_max]
        
        batch_size = images.size(0)
        max_len = targets.size(1)
        device = images.device
        
        encoder_out = self.encoder(images)
        seq_len = encoder_out.size(1)
        hidden, hidden2 = self._init_hidden(encoder_out)
        
        outputs = torch.zeros(batch_size, max_len, self.vocab_size, device=device)
        coverage = torch.zeros(batch_size, seq_len, device=device)
        coverage_loss = torch.zeros(1, device=device)
        
        decoder_input = targets[:, 0]  # <bos>
        
        for t in range(1, max_len):
            output, hidden, hidden2, alpha = self.decoder(
                decoder_input, hidden, hidden2, encoder_out, coverage
            )
            outputs[:, t, :] = output
            
            # 计算覆盖损失：惩罚重复关注
            coverage_loss = coverage_loss + torch.sum(torch.min(alpha, coverage))
            coverage = coverage + alpha
            
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if use_teacher_forcing:
                decoder_input = targets[:, t]
            else:
                decoder_input = output.argmax(1)
        
        # 归一化覆盖损失
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
        seq_len = encoder_out.size(1)
        hidden, hidden2 = self._init_hidden(encoder_out)
        coverage = torch.zeros(batch_size, seq_len, device=device)
        
        preds = torch.full((batch_size, max_len), bos_id, dtype=torch.long, device=device)
        decoder_input = preds[:, 0]
        
        for t in range(1, max_len):
            output, hidden, hidden2, alpha = self.decoder(
                decoder_input, hidden, hidden2, encoder_out, coverage
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
        beam_width: int = 5,
        length_penalty: float = 0.7,
    ) -> torch.Tensor:
        """Beam Search解码"""
        device = images.device
        batch_size = images.size(0)
        
        all_preds = []
        
        for b in range(batch_size):
            img = images[b:b+1]
            encoder_out = self.encoder(img)
            seq_len = encoder_out.size(1)
            hidden, hidden2 = self._init_hidden(encoder_out)
            coverage = torch.zeros(1, seq_len, device=device)
            
            # 每个beam: (score, sequence, hidden, hidden2, coverage)
            beams = [(0.0, [bos_id], hidden, hidden2, coverage)]
            completed = []
            
            for t in range(1, max_len):
                all_candidates = []
                
                for score, seq, h, h2, cov in beams:
                    if seq[-1] == eos_id:
                        # 长度惩罚
                        final_score = score / (len(seq) ** length_penalty)
                        completed.append((final_score, seq))
                        continue
                    
                    decoder_input = torch.tensor([seq[-1]], dtype=torch.long, device=device)
                    output, new_h, new_h2, alpha = self.decoder(
                        decoder_input, h, h2, encoder_out, cov
                    )
                    new_cov = cov + alpha
                    
                    log_probs = F.log_softmax(output, dim=-1)
                    topk_probs, topk_ids = log_probs.topk(beam_width, dim=-1)
                    
                    for i in range(beam_width):
                        new_score = score + topk_probs[0, i].item()
                        new_seq = seq + [topk_ids[0, i].item()]
                        all_candidates.append((new_score, new_seq, new_h, new_h2, new_cov))
                
                if not all_candidates:
                    break
                
                all_candidates.sort(key=lambda x: x[0], reverse=True)
                beams = all_candidates[:beam_width]
                
                if len(completed) >= beam_width:
                    break
            
            # 添加未完成的beam
            for score, seq, _, _, _ in beams:
                final_score = score / (len(seq) ** length_penalty)
                completed.append((final_score, seq))
            
            if completed:
                completed.sort(key=lambda x: x[0], reverse=True)
                best_seq = completed[0][1]
            else:
                best_seq = [bos_id, eos_id]
            
            if len(best_seq) < max_len:
                best_seq = best_seq + [0] * (max_len - len(best_seq))
            else:
                best_seq = best_seq[:max_len]
            
            all_preds.append(best_seq)
        
        return torch.tensor(all_preds, dtype=torch.long, device=device)


def build_e2e_model_v2(vocab_size: int, cfg: dict) -> E2EModelV2:
    return E2EModelV2(vocab_size, cfg)
