import math
from typing import List, Optional, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS


@MODELS.register_module()
class TPVFormerPositionalEncoding(BaseModule):
    """Positional encoding for TPVFormer.

    Args:
        num_feats (Union[int, List[int]]): Number of features for each dimension.
        h (int): Height of the TPV grid.
        w (int): Width of the TPV grid.
        z (int): Depth of the TPV grid.
        temperature (int): Temperature for the positional encoding.
            Defaults to 10000.
        normalize (bool): Whether to normalize the positional encoding.
            Defaults to False.
        scale (float): Scale factor for the positional encoding.
            Defaults to 1.0.
        eps (float): Epsilon for numerical stability.
            Defaults to 1e-6.
    """

    def __init__(self,
                 num_feats: Union[int, List[int]],
                 h: int,
                 w: int,
                 z: int,
                 temperature: int = 10000,
                 normalize: bool = False,
                 scale: float = 1.0,
                 eps: float = 1e-6) -> None:
        super().__init__()
        if isinstance(num_feats, int):
            num_feats = [num_feats] * 3
        self.num_feats = num_feats
        self.h = h
        self.w = w
        self.z = z
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps

        # Create positional encodings for each dimension
        self.h_embed = nn.Parameter(torch.randn(1, h, num_feats[0]))
        self.w_embed = nn.Parameter(torch.randn(1, w, num_feats[1]))
        self.z_embed = nn.Parameter(torch.randn(1, z, num_feats[2]))

    def forward(self, batch_size: int) -> List[torch.Tensor]:
        """Forward function.

        Args:
            batch_size (int): Batch size.

        Returns:
            List[torch.Tensor]: List of positional encodings for each TPV view.
        """
        # Expand to batch size
        h_embed = self.h_embed.expand(batch_size, -1, -1)
        w_embed = self.w_embed.expand(batch_size, -1, -1)
        z_embed = self.z_embed.expand(batch_size, -1, -1)

        # Create TPV positional encodings
        # H-W view: (B, H*W, C)
        hw_embed = h_embed.unsqueeze(2).expand(-1, -1, self.w, -1)
        hw_embed = hw_embed.reshape(batch_size, self.h * self.w, self.num_feats[0])

        # Z-H view: (B, Z*H, C)
        zh_embed = z_embed.unsqueeze(2).expand(-1, -1, self.h, -1)
        zh_embed = zh_embed.reshape(batch_size, self.z * self.h, self.num_feats[2])

        # W-Z view: (B, W*Z, C)
        wz_embed = w_embed.unsqueeze(2).expand(-1, -1, self.z, -1)
        wz_embed = wz_embed.reshape(batch_size, self.w * self.z, self.num_feats[1])

        return [hw_embed, zh_embed, wz_embed]


@MODELS.register_module()
class LearnedPositionalEncoding(BaseModule):
    """Learned positional encoding for TPVFormer.
    
    This class is designed to be compatible with the original tpv04_occupancy.py
    configuration and structure.
    
    Args:
        num_feats (int): Number of features for positional encoding.
        row_num_embed (int): Number of rows for embedding.
        col_num_embed (int): Number of columns for embedding.
    """

    def __init__(self,
                 num_feats: int,
                 row_num_embed: int,
                 col_num_embed: int) -> None:
        super().__init__()
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed
        
        # Create learnable positional embeddings
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        
        # Initialize embeddings
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the embeddings."""
        nn.init.normal_(self.row_embed.weight, std=0.02)
        nn.init.normal_(self.col_embed.weight, std=0.02)
    
    def forward(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        """Forward function.
        
        Args:
            batch_size (int): Batch size.
            device (torch.device): Device to place tensors on.
            
        Returns:
            List[torch.Tensor]: List of positional encodings for each TPV view.
        """
        # Create position indices
        row_pos = torch.arange(self.row_num_embed, device=device)
        col_pos = torch.arange(self.col_num_embed, device=device)
        
        # Get embeddings
        row_embed = self.row_embed(row_pos)  # (H, C)
        col_embed = self.col_embed(col_pos)  # (W, C)
        
        # Create TPV positional encodings
        # H-W view: (B, H*W, C)
        hw_embed = row_embed.unsqueeze(1).expand(-1, self.col_num_embed, -1)
        hw_embed = hw_embed.reshape(-1, self.num_feats)  # (H*W, C)
        hw_embed = hw_embed.unsqueeze(0).expand(batch_size, -1, -1)  # (B, H*W, C)
        
        # Z-H view: (B, Z*H, C) - Z dimension은 8로 고정
        z_dim = 8  # tpv_z_ 값
        zh_embed = torch.zeros(batch_size, z_dim * self.row_num_embed, 
                              self.num_feats, device=device)
        # 간단한 위치 인코딩 생성
        for i in range(z_dim):
            start_idx = i * self.row_num_embed
            end_idx = (i + 1) * self.row_num_embed
            zh_embed[:, start_idx:end_idx, :] = row_embed.unsqueeze(0).expand(batch_size, -1, -1)
        
        # W-Z view: (B, W*Z, C)
        wz_embed = torch.zeros(batch_size, self.col_num_embed * z_dim, 
                              self.num_feats, device=device)
        # 간단한 위치 인코딩 생성
        for i in range(z_dim):
            start_idx = i * self.col_num_embed
            end_idx = (i + 1) * self.col_num_embed
            wz_embed[:, start_idx:end_idx, :] = col_embed.unsqueeze(0).expand(batch_size, -1, -1)
        
        return [hw_embed, zh_embed, wz_embed]
