# Load compatibility layer first
from . import compat

# Import with error handling for mmcv.runner compatibility
try:
    from .core.evaluation.eval_hooks import CustomDistEvalHook
except ImportError as e:
    print(f"Warning: Could not import CustomDistEvalHook: {e}")
    CustomDistEvalHook = None

try:
    from .models.opt.adamw import AdamW2
except ImportError as e:
    print(f"Warning: Could not import AdamW2: {e}")
    AdamW2 = None

print("[DEBUG __init__.py] Attempting to import ssc_rs...")
try:
    from .ssc_rs import *
    print("[DEBUG __init__.py] ssc_rs imported successfully")
except Exception as e:
    print(f"[ERROR __init__.py] Failed to import ssc_rs: {e}")
    import traceback
    traceback.print_exc()
    raise

print("[DEBUG __init__.py] Attempting to import flashocc...")
try:
    from .flashocc import *
    print("[DEBUG __init__.py] flashocc imported successfully")
except Exception as e:
    print(f"[ERROR __init__.py] Failed to import flashocc: {e}")
    import traceback
    traceback.print_exc()
    raise

print("[DEBUG __init__.py] Attempting to import datasets...")
try:
    from .datasets import *
    print("[DEBUG __init__.py] datasets imported successfully")
except Exception as e:
    print(f"[ERROR __init__.py] Failed to import datasets: {e}")
    import traceback
    traceback.print_exc()
    # Don't raise - datasets might not be critical

# CRITICAL: Copy all registered models and datasets from mmdet3d.registry to mmengine.registry
# This is necessary because mmengine.Runner uses mmengine.registry
print("[DEBUG __init__.py] Synchronizing registries...")
try:
    from mmdet3d.registry import MODELS as MMDET3D_MODELS, DATASETS as MMDET3D_DATASETS
    from mmengine.registry import MODELS as MMENGINE_MODELS, DATASETS as MMENGINE_DATASETS
    
    # Copy all registered models from mmdet3d to mmengine
    models_to_sync = []
    for name, module in MMDET3D_MODELS.module_dict.items():
        if name not in MMENGINE_MODELS.module_dict:
            MMENGINE_MODELS.register_module(name=name, force=False, module=module)
            models_to_sync.append(name)
    
    # Copy all registered datasets from mmdet3d to mmengine
    datasets_to_sync = []
    for name, module in MMDET3D_DATASETS.module_dict.items():
        if name not in MMENGINE_DATASETS.module_dict:
            MMENGINE_DATASETS.register_module(name=name, force=False, module=module)
            datasets_to_sync.append(name)
    
    if models_to_sync:
        print(f"[DEBUG __init__.py] Synced {len(models_to_sync)} models to mmengine.registry: {models_to_sync[:5]}...")
    if datasets_to_sync:
        print(f"[DEBUG __init__.py] Synced {len(datasets_to_sync)} datasets to mmengine.registry: {datasets_to_sync}")
        
except Exception as e:
    print(f"[ERROR __init__.py] Failed to sync registries: {e}")
    import traceback
    traceback.print_exc()
