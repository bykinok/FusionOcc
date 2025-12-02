# projects/SurroundOcc/surroundocc/modules/grid_mask.py

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

class GridMask(nn.Module):
    """Grid mask for data augmentation.
    
    Original implementation from SurroundOcc.
    Adapted for MMEngine compatibility.
    
    Args:
        use_h (bool): Whether to mask horizontally.
        use_w (bool): Whether to mask vertically.
        rotate (int): Maximum rotation angle.
        offset (bool): Whether to apply offset.
        ratio (float): Mask ratio.
        mode (int): Mask mode (0 or 1).
        prob (float): Probability of applying grid mask.
    """
    
    def __init__(self, use_h=True, use_w=True, rotate=1, offset=False, 
                 ratio=0.5, mode=1, prob=0.7):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        
    def set_prob(self, epoch, max_epoch):
        """Set probability based on epoch."""
        self.prob = self.st_prob * epoch / max_epoch
        
    def forward(self, x):
        """Forward function.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
            
        Returns:
            torch.Tensor: Masked tensor.
        """
        # breakpoint()

        # np.random.seed(42)

        if np.random.rand() > self.prob or not self.training:
            return x
            
        n, c, h, w = x.size()
        x = x.view(-1, h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(2, h)
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
                
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0
                
        # Rotate mask using PIL for accurate rotation
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, 
                    (ww - w) // 2:(ww - w) // 2 + w]
        
        # Convert to tensor and match input dtype/device
        mask = torch.from_numpy(mask).to(x.dtype).to(x.device)
        
        if self.mode == 1:
            mask = 1 - mask
            
        mask = mask.expand_as(x)
        
        if self.offset:
            offset = torch.from_numpy(
                2 * (np.random.rand(h, w) - 0.5)
            ).to(x.dtype).to(x.device)
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask
            
        return x.view(n, c, h, w)