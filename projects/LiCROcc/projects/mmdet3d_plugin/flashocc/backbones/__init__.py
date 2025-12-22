# Don't import ResNet from mmdet - it causes mmcv.cnn.MODELS import issues
# Instead, users should register their own ResNet or use CustomResNet

from .resnet import CustomResNet
from .radar_backbone import PtsBackbone

# from .swin import SwinTransformer

__all__ = ['CustomResNet', 'PtsBackbone']
