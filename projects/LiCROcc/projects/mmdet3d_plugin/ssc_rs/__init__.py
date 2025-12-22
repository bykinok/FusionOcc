
print("[DEBUG ssc_rs.__init__] Attempting to import detectors...")
try:
    from .detectors import *
    print("[DEBUG ssc_rs.__init__] detectors imported successfully")
except Exception as e:
    print(f"[ERROR ssc_rs.__init__] Failed to import detectors: {e}")
    import traceback
    traceback.print_exc()
    raise

print("[DEBUG ssc_rs.__init__] Attempting to import modules...")
try:
    from .modules import *
    print("[DEBUG ssc_rs.__init__] modules imported successfully")
except Exception as e:
    print(f"[ERROR ssc_rs.__init__] Failed to import modules: {e}")
    import traceback
    traceback.print_exc()
    raise

# Runner, hooks, utils are only needed for training, make them optional
try:
    from .runner import *
except ImportError as e:
    print(f"Warning: Could not import ssc_rs.runner: {e}")

try:
    from .hooks import *
except ImportError as e:
    print(f"Warning: Could not import ssc_rs.hooks: {e}")

try:
    from .utils import *
except ImportError as e:
    print(f"Warning: Could not import ssc_rs.utils: {e}")

# APIs are optional
try:
    from .apis import *
except ImportError as e:
    print(f"Warning: Could not import ssc_rs.apis: {e}")
