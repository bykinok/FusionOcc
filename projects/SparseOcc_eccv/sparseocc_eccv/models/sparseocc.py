import queue
import numpy as np
import torch
# 구버전 mmcv/mmdet → compat 경유
from ..compat import (
    DETECTORS, get_dist_info, cast_tensor_type, force_fp32, auto_fp16
)
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from .utils import GridMask, pad_multiple, GpuPhotoMetricDistortion


@DETECTORS.register_module()
class SparseOcc(MVXTwoStageDetector):
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 data_aug=None,
                 use_mask_camera=False,
                 depth_supervision=None,
                 init_cfg=None,
                 data_preprocessor=None,
                 **kwargs):

        # 새 mmdet3d MVXTwoStageDetector는 pts_voxel_layer를 받지 않으므로 제거
        # pretrained도 init_cfg로 이관 (하지만 여기서는 단순히 무시)
        super(SparseOcc, self).__init__(
            pts_voxel_encoder=pts_voxel_encoder,
            pts_middle_encoder=pts_middle_encoder,
            pts_fusion_layer=pts_fusion_layer,
            img_backbone=img_backbone,
            pts_backbone=pts_backbone,
            img_neck=img_neck,
            pts_neck=pts_neck,
            pts_bbox_head=pts_bbox_head,
            img_roi_head=img_roi_head,
            img_rpn_head=img_rpn_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor,
        )

        self.grid_mask = GridMask(ratio=0.5, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.use_mask_camera = use_mask_camera
        self.fp16_enabled = False
        self.data_aug = data_aug
        self.color_aug = GpuPhotoMetricDistortion()

        # Auxiliary depth supervision head (선택적)
        self.depth_head = None
        self.depth_feature_level = 1
        if depth_supervision is not None and depth_supervision.get('enabled', False):
            from mmdet3d.registry import MODELS as _MODELS
            self.depth_feature_level = depth_supervision.get('feature_level', 1)
            ds_cfg = {k: v for k, v in depth_supervision.items()
                      if k not in ('enabled', 'feature_level')}
            self.depth_head = _MODELS.build(ds_cfg)

        self.memory = {}
        self.queue = queue.Queue()

        # pretrained 가중치 로드 (구버전 호환)
        if pretrained is not None:
            self._load_pretrained(pretrained)

    def _load_pretrained(self, pretrained):
        """구버전 pretrained 파라미터 처리."""
        import os
        from mmengine.runner import load_checkpoint
        if os.path.isfile(pretrained):
            load_checkpoint(self, pretrained, strict=False)

    @auto_fp16(apply_to=('img',), out_fp32=True)
    def extract_img_feat(self, img):
        if self.use_grid_mask:
            img = self.grid_mask(img)

        img_feats = self.img_backbone(img)

        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        return img_feats

    @auto_fp16(apply_to=('img',))
    def extract_feat(self, img, img_metas=None, **kwargs):
        """Extract features from images and points."""
        if len(img.shape) == 6:
            img = img.flatten(1, 2)  # [B, TN, C, H, W]

        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        img = img.float()

        if self.data_aug is not None:
            if 'img_color_aug' in self.data_aug and self.data_aug['img_color_aug'] and self.training:
                img = self.color_aug(img)

            if 'img_norm_cfg' in self.data_aug:
                img_norm_cfg = self.data_aug['img_norm_cfg']

                norm_mean = torch.tensor(img_norm_cfg['mean'], device=img.device)
                norm_std = torch.tensor(img_norm_cfg['std'], device=img.device)

                if img_norm_cfg['to_rgb']:
                    img = img[:, [2, 1, 0], :, :]  # BGR to RGB

                img = img - norm_mean.reshape(1, 3, 1, 1)
                img = img / norm_std.reshape(1, 3, 1, 1)

            for b in range(B):
                img_shape = (img.shape[2], img.shape[3], img.shape[1])
                img_metas[b]['img_shape'] = [img_shape for _ in range(N)]
                img_metas[b]['ori_shape'] = [img_shape for _ in range(N)]

            if 'img_pad_cfg' in self.data_aug:
                img_pad_cfg = self.data_aug['img_pad_cfg']
                img = pad_multiple(img, img_metas, size_divisor=img_pad_cfg['size_divisor'])
                H, W = img.shape[-2:]

        input_shape = img.shape[-2:]
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        img_feats = self.extract_img_feat(img)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped

    def forward_pts_train(self, mlvl_feats, voxel_semantics, voxel_instances, instance_class_ids, mask_camera, img_metas, gt_depth=None):
        outs = self.pts_bbox_head(mlvl_feats, img_metas)
        loss_inputs = [voxel_semantics, voxel_instances, instance_class_ids, outs]
        # use_mask_camera=True일 때만 loss에 mask_camera 전달
        if self.use_mask_camera and mask_camera is not None:
            losses = self.pts_bbox_head.loss(*loss_inputs, mask_camera=mask_camera)
        else:
            losses = self.pts_bbox_head.loss(*loss_inputs)

        # Auxiliary depth supervision
        if self.depth_head is not None and gt_depth is not None:
            # mlvl_feats[level]: [B, T*N_cam, C, H, W] → 현재 프레임 6 카메라만 추출
            curr_feats = mlvl_feats[self.depth_feature_level][:, :6, :, :, :]  # [B, 6, C, H, W]
            depth_pred = self.depth_head(curr_feats)   # [B, 6, D, H, W]
            losses['loss_depth'] = self.depth_head.get_depth_loss(gt_depth, depth_pred)

        return losses

    # -----------------------------------------------------------------------
    # DC 객체 언래핑 헬퍼
    # -----------------------------------------------------------------------
    @staticmethod
    def _unwrap_dc(val):
        """DataContainer(DC) 래핑 해제. 재귀적으로 처리.
        torch.Tensor의 .data 속성과 혼동하지 않도록 DC 클래스만 처리한다.
        """
        # DC 클래스 여부 확인 (torch.Tensor, np.ndarray 제외)
        if (not isinstance(val, (torch.Tensor, np.ndarray))
                and hasattr(val, 'data')
                and type(val).__name__ == 'DC'):
            return SparseOcc._unwrap_dc(val.data)
        if isinstance(val, list):
            return [SparseOcc._unwrap_dc(v) for v in val]
        return val

    # -----------------------------------------------------------------------
    # forward 인터페이스 (구버전 return_loss + 새 mmengine mode 모두 지원)
    # -----------------------------------------------------------------------
    def forward(self, inputs=None, data_samples=None, mode='tensor',
                return_loss=None, **kwargs):
        # mmengine 새 API: mode 기반 dispatch
        if mode == 'loss':
            # 새 API: inputs가 dict로 왔을 때
            if inputs is not None:
                return self.loss(inputs, data_samples)
            # 구버전 데이터: img, img_metas 등이 DC 래핑으로 kwargs에 올 때
            img = self._unwrap_dc(kwargs.get('img'))
            img_metas = self._unwrap_dc(kwargs.get('img_metas', []))
            if isinstance(img_metas, dict):
                img_metas = [img_metas]
            # img 처리: DC unwrap 후 스택 or 그대로
            if isinstance(img, list):
                try:
                    img = torch.stack([
                        x if isinstance(x, torch.Tensor) else torch.from_numpy(x)
                        for x in img
                    ])
                except Exception:
                    pass
            if isinstance(img, torch.Tensor):
                device = next(self.parameters()).device
                img = img.to(device)
            # gt 데이터 언래핑 + 리스트 텐서 스택
            device = next(self.parameters()).device

            def _unwrap_and_stack(val):
                """DC 언래핑 후 list-of-tensor를 단일 배치 텐서로 스택."""
                val = self._unwrap_dc(val)
                if isinstance(val, list) and len(val) > 0:
                    try:
                        tensors = []
                        for x in val:
                            if isinstance(x, torch.Tensor):
                                tensors.append(x)
                            elif isinstance(x, np.ndarray):
                                tensors.append(torch.from_numpy(np.ascontiguousarray(x)))
                            else:
                                tensors.append(torch.tensor(x))
                        val = torch.stack(tensors, dim=0).to(device)
                    except Exception:
                        pass
                elif isinstance(val, torch.Tensor):
                    val = val.to(device)
                return val

            voxel_semantics = _unwrap_and_stack(kwargs.get('voxel_semantics'))
            voxel_instances = _unwrap_and_stack(kwargs.get('voxel_instances'))
            mask_camera = _unwrap_and_stack(kwargs.get('mask_camera'))
            gt_depth = _unwrap_and_stack(kwargs.get('gt_depth'))
            # instance_class_ids는 배치 아이템마다 길이가 다를 수 있고
            # loss_single 에서 리스트로 재할당하므로 반드시 리스트로 유지
            raw_icids = self._unwrap_dc(kwargs.get('instance_class_ids'))
            if isinstance(raw_icids, list):
                instance_class_ids = [
                    (x.to(device) if isinstance(x, torch.Tensor) else
                     torch.from_numpy(np.ascontiguousarray(x)).to(device)
                     if isinstance(x, np.ndarray) else x)
                    for x in raw_icids
                ]
            elif isinstance(raw_icids, torch.Tensor):
                instance_class_ids = [raw_icids[b].to(device) for b in range(raw_icids.shape[0])]
            else:
                instance_class_ids = raw_icids
            return self.forward_train(
                img_metas=img_metas,
                img=img,
                voxel_semantics=voxel_semantics,
                voxel_instances=voxel_instances,
                instance_class_ids=instance_class_ids,
                mask_camera=mask_camera,
                gt_depth=gt_depth,
            )
        elif mode == 'predict':
            # 새 API: inputs/data_samples 형식
            if inputs is not None:
                return self.predict(inputs, data_samples)
            # 구버전 데이터: img, img_metas 등이 DC 래핑으로 kwargs에 올 때
            img = self._unwrap_dc(kwargs.get('img'))
            img_metas = self._unwrap_dc(kwargs.get('img_metas', []))
            # img 처리: list → tensor [B, N, C, H, W]
            if isinstance(img, list):
                img = torch.stack([
                    x if isinstance(x, torch.Tensor) else torch.from_numpy(x)
                    for x in img
                ])
            # 모델 디바이스로 이동 (CPU → GPU)
            if isinstance(img, torch.Tensor):
                device = next(self.parameters()).device
                img = img.to(device)
            # img_metas 처리: list of dicts 보장
            if isinstance(img_metas, dict):
                img_metas = [img_metas]
            return self.forward_test(img_metas, img)
        elif mode == 'tensor':
            # 구버전 호환: return_loss 기반 dispatch
            if return_loss is not None:
                if return_loss:
                    return self.forward_train(**kwargs)
                else:
                    return self.forward_test(**kwargs)
            return self._forward(inputs, data_samples, **kwargs)
        # fallback: 구버전 API
        if return_loss is not None:
            if return_loss:
                return self.forward_train(**kwargs)
            else:
                return self.forward_test(**kwargs)
        return self.predict(inputs, data_samples)

    @force_fp32(apply_to=('img',))
    def forward_train(self, img_metas=None, img=None, voxel_semantics=None, voxel_instances=None, instance_class_ids=None, mask_camera=None, gt_depth=None, **kwargs):
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        return self.forward_pts_train(img_feats, voxel_semantics, voxel_instances, instance_class_ids, mask_camera, img_metas, gt_depth=gt_depth)

    def forward_test(self, img_metas, img=None, **kwargs):
        from ..datasets.nuscenes_occ_dataset import _get_occ_class_names
        from .utils import sparse2dense

        output = self.simple_test(img_metas, img)

        sem_pred = output['sem_pred'].cpu().numpy().astype(np.uint8)
        occ_loc = output['occ_loc'].cpu().numpy().astype(np.uint8)

        batch_size = sem_pred.shape[0]

        occ_size = list(self.pts_bbox_head.occ_size)
        occ_class_names = self.pts_bbox_head.class_names
        free_id = len(occ_class_names) - 1

        results = []
        for b in range(batch_size):
            sem_b = torch.from_numpy(sem_pred[b:b+1])
            loc_b = torch.from_numpy(occ_loc[b:b+1].astype(np.int64))
            dense_sem, _ = sparse2dense(loc_b, sem_b, dense_shape=occ_size, empty_value=free_id)
            dense_sem_np = dense_sem.squeeze(0).numpy().astype(np.uint8)

            index = img_metas[b].get('index', b) if b < len(img_metas) else b

            result = {
                'occ_results': dense_sem_np,
                'index': index,
                'sem_pred': sem_pred[b:b+1],
                'occ_loc': occ_loc[b:b+1],
            }

            if 'pano_inst' in output and 'pano_sem' in output:
                result['pano_inst'] = output['pano_inst'].cpu().numpy().astype(np.int16)[b:b+1]
                result['pano_sem'] = output['pano_sem'].cpu().numpy().astype(np.uint8)[b:b+1]

            results.append(result)

        return results

    def simple_test_pts(self, x, img_metas, rescale=False):
        outs = self.pts_bbox_head(x, img_metas)
        outs = self.pts_bbox_head.merge_occ_pred(outs)
        return outs

    def simple_test(self, img_metas, img=None, rescale=False):
        world_size = get_dist_info()[1]
        if world_size == 1:  # online
            return self.simple_test_online(img_metas, img, rescale)
        else:  # offline
            return self.simple_test_offline(img_metas, img, rescale)

    def simple_test_offline(self, img_metas, img=None, rescale=False):
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        return self.simple_test_pts(img_feats, img_metas, rescale=rescale)

    def simple_test_online(self, img_metas, img=None, rescale=False):
        self.fp16_enabled = False
        assert len(img_metas) == 1  # batch_size = 1

        B, N, C, H, W = img.shape
        img = img.reshape(B, N//6, 6, C, H, W)

        img_filenames = img_metas[0]['filename']
        num_frames = len(img_filenames) // 6

        img_shape = (H, W, C)
        img_metas[0]['img_shape'] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]['ori_shape'] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]['pad_shape'] = [img_shape for _ in range(len(img_filenames))]

        img_feats_list, img_metas_list = [], []

        for i in range(num_frames):
            img_indices = list(np.arange(i * 6, (i + 1) * 6))

            img_metas_curr = [{}]
            for k in img_metas[0].keys():
                if isinstance(img_metas[0][k], list):
                    img_metas_curr[0][k] = [img_metas[0][k][i] for i in img_indices]

            if img_filenames[img_indices[0]] in self.memory:
                img_feats_curr = self.memory[img_filenames[img_indices[0]]]
            else:
                img_feats_curr = self.extract_feat(img[:, i], img_metas_curr)
                self.memory[img_filenames[img_indices[0]]] = img_feats_curr
                self.queue.put(img_filenames[img_indices[0]])
                while self.queue.qsize() > 16:  # avoid OOM
                    pop_key = self.queue.get()
                    self.memory.pop(pop_key)

            img_feats_list.append(img_feats_curr)
            img_metas_list.append(img_metas_curr)

        feat_levels = len(img_feats_list[0])
        img_feats_reorganized = []
        for j in range(feat_levels):
            feat_l = torch.cat([img_feats_list[i][j] for i in range(len(img_feats_list))], dim=0)
            feat_l = feat_l.flatten(0, 1)[None, ...]
            img_feats_reorganized.append(feat_l)

        img_metas_reorganized = img_metas_list[0]
        for i in range(1, len(img_metas_list)):
            for k, v in img_metas_list[i][0].items():
                if isinstance(v, list):
                    img_metas_reorganized[0][k].extend(v)

        img_feats = img_feats_reorganized
        img_metas = img_metas_reorganized
        img_feats = cast_tensor_type(img_feats, torch.half, torch.float32)

        return self.simple_test_pts(img_feats, img_metas, rescale=rescale)

    # -----------------------------------------------------------------------
    # 새 mmdet3d 1.x API 어댑터
    # Base3DDetector는 loss() / predict() / _forward() 기반으로 동작함
    # 원본 forward_train / forward_test 로직을 래핑한다.
    # -----------------------------------------------------------------------

    def _forward(self, batch_inputs_dict=None, batch_data_samples=None, **kwargs):
        """새 API: 단순 forward pass (추론 전용)."""
        if batch_data_samples is not None:
            img_metas = [s.metainfo for s in batch_data_samples]
            img = batch_inputs_dict.get('imgs', None)
            return self.forward_test(img_metas, img)
        return {}

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """새 mmdet3d API: 학습 시 호출.

        batch_inputs_dict: {'imgs': tensor, ...}
        batch_data_samples: List[Det3DDataSample]
            - metainfo: img_metas
            - gt_fields: voxel_semantics, voxel_instances, instance_class_ids, mask_camera
        """
        img = batch_inputs_dict.get('imgs', None)
        img_metas = [s.metainfo for s in batch_data_samples]

        # gt 데이터 수집
        voxel_semantics = torch.stack([s.gt_fields.voxel_semantics for s in batch_data_samples])
        voxel_instances = torch.stack([s.gt_fields.voxel_instances for s in batch_data_samples])
        instance_class_ids = [s.gt_fields.instance_class_ids for s in batch_data_samples]
        mask_camera = None
        if hasattr(batch_data_samples[0].gt_fields, 'mask_camera'):
            mask_camera = torch.stack([s.gt_fields.mask_camera for s in batch_data_samples])

        gt_depth = None
        if 'gt_depth' in batch_inputs_dict:
            gt_depth = batch_inputs_dict['gt_depth']
        elif hasattr(batch_data_samples[0].gt_fields, 'gt_depth'):
            gt_depth = torch.stack([s.gt_fields.gt_depth for s in batch_data_samples])

        return self.forward_train(
            img_metas=img_metas,
            img=img,
            voxel_semantics=voxel_semantics,
            voxel_instances=voxel_instances,
            instance_class_ids=instance_class_ids,
            mask_camera=mask_camera,
            gt_depth=gt_depth,
        )

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """새 mmdet3d API: 추론 시 호출."""
        img = batch_inputs_dict.get('imgs', None)
        img_metas = [s.metainfo for s in batch_data_samples]

        results = self.forward_test(img_metas, img)

        # Det3DDataSample에 결과 저장 (OccupancyMetricHybrid.process()가 직접 읽음)
        for i, (result, data_sample) in enumerate(zip(results, batch_data_samples)):
            # BaseDataElement.set_field로 data field에 등록
            data_sample.set_field(
                value=[result['occ_results']],
                name='occ_results',
                field_type='data',
            )
            data_sample.set_field(
                value=result['index'],
                name='index',
                field_type='data',
            )

        return batch_data_samples
