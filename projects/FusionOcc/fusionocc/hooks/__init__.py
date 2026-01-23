# Copyright (c) OpenMMLab. All rights reserved.
"""
Custom hooks for FusionOcc.
"""

from .syncbn_hook import SyncBNHook, SyncbnControlHook

__all__ = ['SyncBNHook', 'SyncbnControlHook']
