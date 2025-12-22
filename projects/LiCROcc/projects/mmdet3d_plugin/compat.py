"""
Compatibility layer for mmcv 1.x -> mmengine 2.x migration
"""
import sys
from types import ModuleType
import torch.nn as nn

# mmcv.runner compatibility
try:
    from mmcv.runner import auto_fp16, force_fp32, BaseModule, get_dist_info
except ImportError:
    # mmengine doesn't use these decorators, create dummy ones
    print("Warning: mmcv.runner not found, using compatibility shims")
    
    def auto_fp16(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])
    
    def force_fp32(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])
    
    def get_dist_info():
        """Get distributed info (rank, world_size)"""
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank(), dist.get_world_size()
        except:
            pass
        return 0, 1
    
    # BaseModule compatibility - make it accept init_cfg parameter
    class BaseModule(nn.Module):
        """Compatibility BaseModule that accepts init_cfg"""
        def __init__(self, init_cfg=None):
            super().__init__()
            self.init_cfg = init_cfg
    
    # Create a fake mmcv.runner module and inject it into sys.modules
    # This allows existing imports to work
    if 'mmcv' not in sys.modules:
        mmcv_module = ModuleType('mmcv')
        sys.modules['mmcv'] = mmcv_module
    
    if 'mmcv.runner' not in sys.modules:
        runner_module = ModuleType('mmcv.runner')
        runner_module.auto_fp16 = auto_fp16
        runner_module.force_fp32 = force_fp32
        runner_module.BaseModule = BaseModule
        runner_module.get_dist_info = get_dist_info
        sys.modules['mmcv.runner'] = runner_module
        sys.modules['mmcv'].runner = runner_module

# mmcv.cnn initialization functions compatibility
try:
    from mmcv.cnn import constant_init, xavier_init, normal_init, uniform_init, kaiming_init, bias_init_with_prob
except ImportError:
    print("Warning: mmcv.cnn init functions not found, using torch.nn.init fallbacks")
    
    def constant_init(module, val, bias=0):
        """Initialize module with constant value."""
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)
    
    def xavier_init(module, gain=1, bias=0, distribution='normal'):
        """Initialize module with Xavier initialization."""
        if hasattr(module, 'weight') and module.weight is not None:
            if distribution == 'uniform':
                nn.init.xavier_uniform_(module.weight, gain=gain)
            else:
                nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)
    
    def normal_init(module, mean=0, std=1, bias=0):
        """Initialize module with normal distribution."""
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)
    
    def uniform_init(module, a=0, b=1, bias=0):
        """Initialize module with uniform distribution."""
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.uniform_(module.weight, a, b)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)
    
    def kaiming_init(module, a=0, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal'):
        """Initialize module with Kaiming initialization."""
        if hasattr(module, 'weight') and module.weight is not None:
            if distribution == 'uniform':
                nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
            else:
                nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)
    
    def bias_init_with_prob(prior_prob=0.01):
        """Initialize bias with probability."""
        import math
        bias_init = float(-math.log((1 - prior_prob) / prior_prob))
        return bias_init
    
    # Inject into mmcv.cnn module
    # mmcv 2.x is installed but doesn't have these init functions
    # We need to add them to the existing mmcv.cnn module
    try:
        import mmcv.cnn as mmcv_cnn
        # Add missing functions to the existing module
        if not hasattr(mmcv_cnn, 'constant_init'):
            mmcv_cnn.constant_init = constant_init
        if not hasattr(mmcv_cnn, 'xavier_init'):
            mmcv_cnn.xavier_init = xavier_init
        if not hasattr(mmcv_cnn, 'normal_init'):
            mmcv_cnn.normal_init = normal_init
        if not hasattr(mmcv_cnn, 'uniform_init'):
            mmcv_cnn.uniform_init = uniform_init
        if not hasattr(mmcv_cnn, 'kaiming_init'):
            mmcv_cnn.kaiming_init = kaiming_init
        if not hasattr(mmcv_cnn, 'bias_init_with_prob'):
            mmcv_cnn.bias_init_with_prob = bias_init_with_prob
    except ImportError:
        # mmcv not installed at all, create fake module
        if 'mmcv' not in sys.modules:
            mmcv_module = ModuleType('mmcv')
            sys.modules['mmcv'] = mmcv_module
        
        if 'mmcv.cnn' not in sys.modules:
            cnn_module = ModuleType('mmcv.cnn')
            cnn_module.constant_init = constant_init
            cnn_module.xavier_init = xavier_init
            cnn_module.normal_init = normal_init
            cnn_module.uniform_init = uniform_init
            cnn_module.kaiming_init = kaiming_init
            cnn_module.bias_init_with_prob = bias_init_with_prob
            sys.modules['mmcv.cnn'] = cnn_module
            sys.modules['mmcv'].cnn = cnn_module

__all__ = [
    'auto_fp16', 'force_fp32', 'BaseModule', 'get_dist_info',
    'constant_init', 'xavier_init', 'normal_init', 'uniform_init', 
    'kaiming_init', 'bias_init_with_prob'
]

