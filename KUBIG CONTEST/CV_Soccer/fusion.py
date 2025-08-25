import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

import random

# include in configs.py

# feature_dim = 512 # referred to as D in footnote
# num_heads = 8
# dropout = 0.2

class CrossModalGatedAttention(nn.Module):
  def __init__(self, feature_dim=512, num_heads=8, dropout=0.2):
    super(CrossModalGatedAttention, self).__init__()

    self.num_heads = num_heads
    self.head_dim = feature_dim // num_heads
    self.feature_dim = feature_dim

    self.scale = self.feature_dim ** -0.5
    self.Scale = self.head_dim ** -0.5

    assert feature_dim % num_heads == 0

    # cross-modality attention
    self.qkv_i = nn.Linear(feature_dim, feature_dim*3, bias = False)
    self.qkv_j = nn.Linear(feature_dim,feature_dim*3, bias = False)
 #   self.proj = nn.Linear(feature_dim, feature_dim) <- not used
    self.att_drop = nn.Dropout(dropout)
 #   self.proj_drop = nn.Dropout(dropout) <- not used
    self.sigmoid = nn.Sigmoid()
    self.relu = nn.ReLU()

    # forget gate weights
    self.W_f = nn.Linear(feature_dim*2, feature_dim, bias=True)
    self.W_m = nn.Linear(feature_dim, feature_dim, bias=True)

    # fusion layer attention weights
    self.QKV = nn.Linear(feature_dim, feature_dim*3, bias = False)
    self.Proj = nn.Linear(feature_dim, feature_dim)
    self.Att_drop = nn.Dropout(dropout)
    self.Proj_drop = nn.Dropout(dropout)

    # initialize weights
    self._init_weights()

  def _init_weights(self):
    nn.init.xavier_uniform_(self.qkv_i.weight)
    nn.init.xavier_uniform_(self.qkv_j.weight)
    nn.init.xavier_uniform_(self.QKV.weight)
#    nn.init.xavier_uniform_(self.proj.weight) <- not used
    nn.init.xavier_uniform_(self.Proj.weight)
    nn.init.xavier_uniform_(self.W_f.weight)
    nn.init.xavier_uniform_(self.W_m.weight)
#    nn.init.constant_(self.proj.bias, 0) <- not used
    if self.Proj.bias is not None:
      nn.init.constant_(self.Proj.bias, 0)

  @staticmethod
  def _ensure_3d(x):
    if x.dim() == 2:
      return x.unsqueeze(1)
    elif x.dim() == 3:
      return x
    raise ValueError("input must be (B x D) or (B x N x D)")

  def forward(self, x_1, x_2):
    z_i, z_j = self._ensure_3d(x_1), self._ensure_3d(x_2)

    # each input: (B x N x D)
    # output:  (B x N x D)
    B_1, N_1, F_1 = z_i.shape
    B_2, N_2, F_2 = z_j.shape
    if B_1 != B_2:
      raise ValueError("batch size does not match for two modalities")
    if N_1 != N_2:
      raise ValueError("the number of frames does not match for two modalities")
    if F_1 != F_2:
      raise ValueError("the number of features does not match for two modalities")

    batch_size = B_1
    num_frames = N_1
    feature_dim = F_1

    # (1) cross-attention > forget gate

    qkv_i = self.qkv_i(z_i)# (B x N x 3D)
    qkv_j = self.qkv_j(z_j) # (B x N x 3D)
    qkv_i = qkv_i.reshape(batch_size, num_frames, 3, feature_dim) # (B x N x 3 x D)
    qkv_j = qkv_j.reshape(batch_size, num_frames, 3, feature_dim) # (B x N x 3 x D)

    _, k_i, v_i = qkv_i.permute(2, 0, 1, 3) # each: (B x N x D)
    q_j, _, _ = qkv_j.permute(2, 0, 1, 3) # each: (B x N x D)

      # (video, audio) pair # (audio, video)의 reverse order는 생략 bcs performance 차이 insignificant according to the paper
    a_ij = F.softmax((q_j @ k_i.transpose(-2, -1) * self.scale), dim=-1)  # (B x N x N)
    a_ij = self.att_drop(a_ij) @ v_i # (B x N x D)
    f_ij = self.sigmoid(self.W_f(torch.concat((a_ij, z_j), dim = 2))) # (B x N x D)
    h_ij = self.relu(z_i + (self.W_m(a_ij) *f_ij)) # (B x N x D)

    # (2) Fusion Layer
    QKV = self.QKV(h_ij) # (B x N x 3D)
    QKV = QKV.reshape(batch_size, num_frames, 3, self.num_heads, self.head_dim) # (B x N x 3 x Nh x Dh) : D -> Nh x Dh
    Q, K, V = QKV.permute(2, 0, 3, 1, 4) # each: (B x Nh x N x Dh)

    Attn = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) * self.Scale, dim=-1) # (B x Nh x N x N)
    H = self.Att_drop(Attn) @ V # (B x Nh x N x Dh)
    H = H.transpose(1, 2).reshape(batch_size, num_frames, -1) # (B x N x D): Nh x Dh > D)
    H = self.Proj(H) # (B x N x D)
    H = self.Proj_drop(H) # (B x N x D)

    return H



# Early fusion에만 의미 있음

class CrossModalGatedBottleneckAttention(nn.Module):
  def __init__(self, feature_dim=512, num_heads=8, dropout=0.4):
    super(CrossModalGatedBottleneckAttention, self).__init__()

    self.num_heads = num_heads
    self.head_dim = feature_dim // num_heads
    self.feature_dim = feature_dim

    self.scale = self.feature_dim ** -0.5
    self.Scale = self.head_dim ** -0.5

    assert feature_dim % num_heads == 0

    # cross-modality attention
      # each modality
    self.qkv_i = nn.Linear(feature_dim, feature_dim*3, bias = False)
    self.qkv_j = nn.Linear(feature_dim,feature_dim*3, bias = False)
    self.att_drop = nn.Dropout(dropout)
    self.sigmoid = nn.Sigmoid()
    self.relu = nn.ReLU()

      # bottleneck
    self.qkv_b = nn.Linear(feature_dim, feature_dim*3, bias = False)

    # forget gate
    self.W_f = nn.Linear(feature_dim*2, feature_dim, bias=True)
    self.W_m = nn.Linear(feature_dim, feature_dim, bias=True)

    # fusion layer attention
    self.QKV = nn.Linear(feature_dim, feature_dim*3, bias = False)
    self.Proj = nn.Linear(feature_dim, feature_dim)
    self.Att_drop = nn.Dropout(dropout)
    self.Proj_drop = nn.Dropout(dropout)


    # initialize weights
    self._init_weights()

  def _init_weights(self):
    nn.init.xavier_uniform_(self.qkv_i.weight)
    nn.init.xavier_uniform_(self.qkv_j.weight)
    nn.init.xavier_uniform_(self.qkv_b.weight)
    nn.init.xavier_uniform_(self.QKV.weight)
#    nn.init.xavier_uniform_(self.proj.weight) <- not used
    nn.init.xavier_uniform_(self.Proj.weight)
    nn.init.xavier_uniform_(self.W_f.weight)
    nn.init.xavier_uniform_(self.W_m.weight)
#    nn.init.constant_(self.proj.bias, 0) <- not used
    if self.Proj.bias is not None:
      nn.init.constant_(self.Proj.bias, 0)

  @staticmethod
  def _ensure_3d(x):
    if x.dim() == 2:
      return x.unsqueeze(1)
    elif x.dim() == 3:
      return x
    raise ValueError("input must be (B x D) or (B x N x D)")

  def forward(self, x_1, x_2):
    z_i, z_j = self._ensure_3d(x_1), self._ensure_3d(x_2)

    # each input: (B x N x D)
    # output:  (B x N x D)
    B_1, N_1, F_1 = z_i.shape
    B_2, N_2, F_2 = z_j.shape
    if B_1 != B_2:
      raise ValueError("batch size does not match for two modalities")
    if N_1 != N_2:
      raise ValueError("the number of frames does not match for two modalities")
    if F_1 != F_2:
      raise ValueError("the number of features does not match for two modalities")

    batch_size = B_1
    num_frames = N_1
    bneck_frames = N_1 // 3
    feature_dim = F_1

    z_b = torch.randn(bneck_frames, feature_dim, device=z_i.device)*0.2
    z_b = z_b.unsqueeze(0).expand(batch_size, -1, -1)
    # (1) cross-attention > forget gate

    qkv_i = self.qkv_i(z_i)# (B x N x 3D)
    qkv_j = self.qkv_j(z_j) # (B x N x 3D)
    qkv_b = self.qkv_b(z_b) # (B x bN x 3D)

    qkv_i = qkv_i.reshape(batch_size, num_frames, 3, feature_dim) # (B x N x 3 x D)
    qkv_j = qkv_j.reshape(batch_size, num_frames, 3, feature_dim) # (B x N x 3 x D)
    qkv_b = qkv_b.reshape(batch_size, bneck_frames, 3, feature_dim) # (B x bN x 3 x D)

    _, k_i, v_i = qkv_i.permute(2, 0, 1, 3) # each: (B x N x D)
    q_j, _, _ = qkv_j.permute(2, 0, 1, 3) # each: (B x N x D)
    q_b, k_b, v_b = qkv_b.permute(2, 0, 1, 3) # each: (B x bN x D)

      # (video, audio) pair # (audio, video)의 reverse order는 생략 bcs performance 차이 insignificant according to the paper
      # force each modality(i and j) to only interact with bottleneck unit # idea from Nagrania et al. (2021) "Attention Bottlenecks for Multimodal Fusion"
    a_ib = F.softmax((q_b @ k_i.transpose(-2, -1) * self.scale), dim=-1)  # (B x bN x N)
    a_bj = F.softmax((q_j @ k_b.transpose(-2, -1) * self.scale), dim=-1) # (B x N x bN)

    a_ib = self.att_drop(a_ib) @ v_i # (B x bN x D)
    attn_upsample = F.softmax((z_i @ a_ib.transpose(-2, -1)) * self.scale, dim=-1) # (B x N x bN)
    a_ib = attn_upsample @ a_ib # (B x N x D) ## how does this become B x N x D?
    a_bj = self.att_drop(a_bj) @ v_b # (B x N x D)

      # average over them <- not following the paper (since averaging is only used to produce Zfsn with Zfsn_a and Zfsn_v)
    a_ij = (a_ib + a_bj) / 2 # element-wise averaging # (B x N x D)
    f_ij = self.sigmoid(self.W_f(torch.concat((a_ij, z_j)), dim=2)) # (B x N x D)
    h_ij = self.relu(z_i + (self.W_m(a_ij) *f_ij)) # (B x N x D)

    # (2) Fusion Layer
    QKV = self.QKV(h_ij) # (B x N x 3D)
    QKV = QKV.reshape(batch_size, num_frames, 3, self.num_heads, self.head_dim) # (B x N x 3 x Nh x Dh) : D -> Nh x Dh
    Q, K, V = QKV.permute(2, 0, 3, 1, 4) # each: (B x Nh x N x Dh)

    Attn = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) * self.Scale, dim=-1) # (B x Nh x N x N)
    H = self.Att_drop(Attn) @ V # (B x Nh x N x Dh)
    H = H.transpose(1, 2).reshape(batch_size, num_frames, -1) # (B x N x D): Nh x Dh > D)
    H = self.Proj(H) # (B x N x D)
    H = self.Proj_drop(H) # (B x N x D)

    return H
  

class GatedBimodalTorch(nn.Module):
    """GMU: h = z ⊙ tanh(Wv x_v) + (1 - z) ⊙ tanh(Wt x_t),
       z = sigmoid(Wz [x_v, x_t]).
       x_v, x_t: [B,D] 또는 [B,T,D]
    """
    def __init__(self, in_dim_v, in_dim_t, hid_dim, bias=True):
        super().__init__()
        self.fv   = nn.Linear(in_dim_v, hid_dim, bias=bias)   # Wv
        self.ft   = nn.Linear(in_dim_t, hid_dim, bias=bias)   # Wt
        self.gate = nn.Linear(in_dim_v + in_dim_t, hid_dim, bias=bias)  # Wz
        # 초기화(선택)
        nn.init.xavier_uniform_(self.fv.weight);  nn.init.zeros_(self.fv.bias)
        nn.init.xavier_uniform_(self.ft.weight);  nn.init.zeros_(self.ft.bias)
        nn.init.xavier_uniform_(self.gate.weight); nn.init.zeros_(self.gate.bias)
    def forward(self, x_v, x_t):                  # [..., Dv], [..., Dt]
        hv = torch.tanh(self.fv(x_v))            # [..., H]
        ht = torch.tanh(self.ft(x_t))            # [..., H]
        z  = torch.sigmoid(self.gate(torch.cat([x_v, x_t], dim=-1)))  # [..., H]
        h  = z * hv + (1.0 - z) * ht             # [..., H]
        return h, z
      
  
class LateFusionTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=2, dim_ff=1024, dropout=0.1, use_cls=True):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.use_cls = use_cls
        if use_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls, std=0.02)
        # 간단한 모달리티 임베딩(영상=0, 오디오=1)
        self.mod_embed = nn.Embedding(2, d_model)
    def forward(self, v_vec, a_vec):
        # v_vec, a_vec: [B, D] (RVLAD 출력)
        B, D = v_vec.shape
        v = v_vec.unsqueeze(1)  # [B,1,D]
        a = a_vec.unsqueeze(1)  # [B,1,D]
        # 모달리티 임베딩 추가
        v = v + self.mod_embed.weight[0].view(1,1,D)
        a = a + self.mod_embed.weight[1].view(1,1,D)
        tokens = torch.cat([v, a], dim=1)  # [B, 2, D]
        if self.use_cls:
            cls = self.cls.expand(B, -1, -1)  # [B,1,D]
            tokens = torch.cat([cls, tokens], dim=1)  # [B,3,D]
        out = self.encoder(tokens)  # [B, 3, D] 또는 [B,2,D]
        fused = out[:, 0, :] if self.use_cls else out.mean(dim=1)  # [B,D]
        return fused

