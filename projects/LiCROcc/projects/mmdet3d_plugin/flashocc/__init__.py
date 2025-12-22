# Import with error handling for compatibility
try:
    from .necks import *
except ImportError as e:
    print(f"Warning: Could not import flashocc.necks: {e}")

try:
    from .backbones import *
except ImportError as e:
    print(f"Warning: Could not import flashocc.backbones: {e}")

try:
    from .dense_heads import *
except ImportError as e:
    print(f"Warning: Could not import flashocc.dense_heads: {e}")

try:
    from .losses import *
except ImportError as e:
    print(f"Warning: Could not import flashocc.losses: {e}")