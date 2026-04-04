from .match_cost import DiceCost, CrossEntropyLossCost

try:
    from mmdet.models.task_modules.assigners.match_cost import (
        ClassificationCost, FocalLossCost)
except ImportError:
    try:
        from mmdet.core.bbox.match_costs import ClassificationCost, FocalLossCost
    except ImportError:
        ClassificationCost = None
        FocalLossCost = None
