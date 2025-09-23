# Grid mask implementation for SurroundOcc
import torch
import torch.nn as nn
import numpy as np


class GridMask(nn.Module):
    """Grid mask for image augmentation.
    
    Args:
        use_h (bool): Whether to use horizontal grid.
        use_w (bool): Whether to use vertical grid.
        rotate (int): Rotation mode.
        offset (bool): Whether to use offset.
        ratio (float): Ratio of grid mask.
        mode (int): Mode of grid mask.
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
        self.prob = prob
        
    def forward(self, x):
        """Forward function."""
        if not self.training or np.random.rand() > self.prob:
            return x
            
        n, c, h, w = x.size()
        x = x.view(-1, h, w)
        
        # Create grid mask
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
                
        r = np.random.randint(self.rotate)
        mask = mask[r:r+h, r:r+w]
        mask = torch.from_numpy(mask).float().cuda()
        
        if self.mode == 1:
            mask = 1 - mask
            
        mask = mask.expand_as(x)
        x = x * mask
        
        return x.view(n, c, h, w)
