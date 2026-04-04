# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
#  Migration: mmseg/mmdet 의존성 → mmengine 기반
# ---------------------------------------------

from .mmdet_train import custom_train_detector


def custom_train_model(model,
                       dataset,
                       cfg,
                       distributed=False,
                       validate=False,
                       timestamp=None,
                       eval_model=None,
                       meta=None):
    """모델 타입에 따라 학습 함수를 분기합니다."""
    custom_train_detector(
        model,
        dataset,
        cfg,
        distributed=distributed,
        validate=validate,
        timestamp=timestamp,
        eval_model=eval_model,
        meta=meta)


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """단순 래퍼: custom_train_model 을 호출합니다."""
    custom_train_model(
        model,
        dataset,
        cfg,
        distributed=distributed,
        validate=validate,
        timestamp=timestamp,
        meta=meta)
