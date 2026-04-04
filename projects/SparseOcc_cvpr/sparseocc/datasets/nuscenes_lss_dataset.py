"""
CustomNuScenesOccLSSDataset — 새 mmdet3d(mmengine BaseDataset) 기반 마이그레이션

원본: Ref/SparseOcc_cvpr_ori/projects/mmdet3d_plugin/datasets/nuscenes_lss_dataset.py
변경점: mmdet3d 구버전(NuScenesDataset 상속) → mmengine BaseDataset 상속
       내부 data loading / evaluation 로직은 원본과 동일하게 유지
"""

import os
import pickle
import numpy as np
from typing import Callable, List, Optional, Union

try:
    from mmengine.dataset import BaseDataset
    from mmengine.fileio import load as mm_load
except ImportError:
    # 설치 전 테스트를 위한 폴백
    BaseDataset = object
    mm_load = None

try:
    from mmdet3d.registry import DATASETS
except ImportError:
    from ..compat import DATASETS

import pdb


@DATASETS.register_module()
class CustomNuScenesOccLSSDataset(BaseDataset):
    """NuScenes Occupancy LSS Dataset (새 mmengine BaseDataset 기반).

    구버전 mmdet3d NuScenesDataset을 상속하던 원본 클래스를 새 API로 이식합니다.
    .pkl annotation 파일 형식은 그대로 유지합니다.

    Args:
        data_root (str): 데이터셋 루트 경로.
        ann_file (str): annotation pkl 파일 경로.
        occ_size (list): voxel occupancy grid 크기 [X, Y, Z].
        pc_range (list): point cloud 범위 [xmin, ymin, zmin, xmax, ymax, zmax].
        pipeline (list): 데이터 전처리 파이프라인.
        classes (list): 클래스 이름 목록.
        modality (dict): 데이터 모달리티 설정.
        test_mode (bool): 테스트 모드 여부.
        use_valid_flag (bool): 유효 플래그 사용 여부.
    """

    METAINFO = {
        'classes': (
            'empty', 'barrier', 'bicycle', 'bus', 'car',
            'construction_vehicle', 'motorcycle', 'pedestrian',
            'traffic_cone', 'trailer', 'truck', 'driveable_surface',
            'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation',
        ),
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 occ_size: list,
                 pc_range: list,
                 pipeline: List[Union[dict, Callable]] = [],
                 classes: Optional[list] = None,
                 modality: Optional[dict] = None,
                 test_mode: bool = False,
                 use_valid_flag: bool = False,
                 **kwargs):

        self.occ_size = occ_size
        self.pc_range = pc_range
        self.use_valid_flag = use_valid_flag
        self.modality = modality or dict(
            use_camera=True, use_lidar=False,
            use_radar=False, use_map=False, use_external=False)

        if classes is not None:
            metainfo = dict(classes=classes)
        else:
            metainfo = None

        # BaseDataset 초기화
        # serialize_data=False: data_list를 직렬화하지 않아 data_list 접근 유지
        # (기본값 True이면 _serialize_data() 후 data_list.clear()로 비워짐 → __len__/get_data_info 오작동)
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            metainfo=metainfo,
            pipeline=pipeline,
            test_mode=test_mode,
            serialize_data=False,
            **kwargs)

        # DistributedGroupSampler 호환을 위한 group flag 설정
        # (모든 샘플을 같은 그룹 0으로 지정: NuScenes는 동일 종횡비)
        self._set_group_flag()

    # ------------------------------------------------------------------
    # BaseDataset 필수 메서드
    # ------------------------------------------------------------------

    def load_data_list(self) -> list:
        """annotation pkl 파일로부터 data_list를 로드합니다."""
        # mmengine BaseDataset._join_prefix()가 이미 data_root + ann_file을 합쳐
        # self.ann_file에 저장한다. 정규화 후 직접 사용.
        ann_path = os.path.normpath(self.ann_file)

        with open(ann_path, 'rb') as f:
            data = pickle.load(f)

        # 구버전 pkl 형식: {'infos': [...], 'metadata': {...}}
        if isinstance(data, dict) and 'infos' in data:
            data_infos = data['infos']
        elif isinstance(data, list):
            data_infos = data
        else:
            raise ValueError(f"Unsupported annotation file format: {type(data)}")

        data_list = []
        for idx, info in enumerate(data_infos):
            if self.use_valid_flag:
                valid_flag = info.get('valid_flag', True)
                # valid_flag가 numpy array일 수 있음 (NuScenes PKL 형식)
                # occupancy 데이터셋은 GT 박스가 없어 빈 배열일 수 있음 → 빈 배열은 유효 샘플로 간주
                if hasattr(valid_flag, '__len__'):
                    if len(valid_flag) > 0 and not bool(
                        valid_flag.any() if hasattr(valid_flag, 'any') else any(valid_flag)
                    ):
                        continue
                elif not valid_flag:
                    continue
            data_list.append(info)

        return data_list

    def get_data_info(self, index: int) -> dict:
        """인덱스에 해당하는 data info를 반환합니다.

        원본 NuScenesLSS 코드와 동일한 입력 dict 구조를 유지합니다.
        """
        info = self.data_list[index]

        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'].replace('./data/nuscenes', self.data_root),
            sweeps=info['sweeps'],
            lidar2ego_translation=info['lidar2ego_translation'],
            lidar2ego_rotation=info['lidar2ego_rotation'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info.get('can_bus', np.zeros(18)),
            frame_idx=info.get('frame_idx', 0),
            timestamp=info['timestamp'],
            occ_size=np.array(self.occ_size),
            pc_range=np.array(self.pc_range),
            lidar_token=info.get('lidar_token', ''),
        )

        if 'scene_name' in info:
            input_dict['scene_name'] = info['scene_name']

        if 'lidarseg' in info:
            input_dict['lidarseg'] = info['lidarseg']

        # 카메라 파일 경로 및 lidar2cam 변환 행렬
        img_filenames = {}
        lidar2cam_dic = {}

        for cam_type, cam_info in info['cams'].items():
            cam_info['data_path'] = cam_info['data_path'].replace(
                './data/nuscenes', self.data_root)
            img_filenames[cam_type] = cam_info['data_path']

            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            lidar2cam_dic[cam_type] = lidar2cam_rt.T

        input_dict['curr'] = info
        input_dict['img_filenames'] = img_filenames
        input_dict['lidar2cam_dic'] = lidar2cam_dic

        return input_dict

    def prepare_data(self, index: int):
        """파이프라인을 적용하여 학습/테스트 데이터를 준비합니다."""
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None

        # 구버전 pre_pipeline 역할 (기본 meta 필드 설정)
        input_dict['img_fields'] = []
        input_dict['bbox3d_fields'] = []
        input_dict['pts_mask_fields'] = []
        input_dict['pts_seg_fields'] = []
        input_dict['bbox_fields'] = []
        input_dict['mask_fields'] = []
        input_dict['seg_fields'] = []

        example = self.pipeline(input_dict)
        return example

    def __getitem__(self, idx: int):
        if self.test_mode:
            return self.prepare_data(idx)

        while True:
            data = self.prepare_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def __len__(self):
        return len(self.data_list)

    def _set_group_flag(self):
        """DistributedGroupSampler 호환을 위한 flag 설정.

        NuScenes는 모든 이미지가 동일 종횡비이므로 flag=0으로 통일합니다.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def _rand_another(self, idx: int) -> int:
        return np.random.randint(0, len(self))

    # ------------------------------------------------------------------
    # 평가 메서드 (원본과 동일)
    # ------------------------------------------------------------------

    def evaluate_lidarseg(self, results, logger=None, **kwargs):
        from ..utils import cm_to_ious, format_results
        eval_results = {}

        ious = cm_to_ious(results['evaluation_semantic'])
        res_table, res_dic = format_results(ious, return_dic=True)
        for key, val in res_dic.items():
            eval_results['nuScenes_lidarseg_{}'.format(key)] = val

        if logger is not None:
            logger.info('LiDAR Segmentation Evaluation')
            logger.info(res_table)

        return eval_results

    def evaluate_ssc(self, results, logger=None, **kwargs):
        eval_results = {}
        if 'ssc_scores' in results:
            ssc_scores = results['ssc_scores']
            class_ssc_iou = ssc_scores['iou_ssc'].tolist()
            res_dic = {
                "SC_Precision": ssc_scores['precision'].item(),
                "SC_Recall": ssc_scores['recall'].item(),
                "SC_IoU": ssc_scores['iou'],
                "SSC_mIoU": ssc_scores['iou_ssc_mean'],
            }
        else:
            assert 'ssc_results' in results
            ssc_results = results['ssc_results']
            completion_tp = sum([x[0] for x in ssc_results])
            completion_fp = sum([x[1] for x in ssc_results])
            completion_fn = sum([x[2] for x in ssc_results])

            tps = sum([x[3] for x in ssc_results])
            fps = sum([x[4] for x in ssc_results])
            fns = sum([x[5] for x in ssc_results])

            precision = completion_tp / (completion_tp + completion_fp)
            recall = completion_tp / (completion_tp + completion_fn)
            iou = completion_tp / (completion_tp + completion_fp + completion_fn)
            iou_ssc = tps / (tps + fps + fns + 1e-5)

            class_ssc_iou = iou_ssc.tolist()
            res_dic = {
                "SC_Precision": precision,
                "SC_Recall": recall,
                "SC_IoU": iou,
                "SSC_mIoU": iou_ssc[1:17].mean(),
            }

            if 'ssc_results_refine' in results:
                ssc_results = results['ssc_results_refine']
                completion_tp = sum([x[0] for x in ssc_results])
                completion_fp = sum([x[1] for x in ssc_results])
                completion_fn = sum([x[2] for x in ssc_results])

                tps = sum([x[3] for x in ssc_results])
                fps = sum([x[4] for x in ssc_results])
                fns = sum([x[5] for x in ssc_results])

                precision = completion_tp / (completion_tp + completion_fp)
                recall = completion_tp / (completion_tp + completion_fn)
                iou = completion_tp / (completion_tp + completion_fp + completion_fn)
                iou_ssc = tps / (tps + fps + fns + 1e-5)

                class_ssc_iou_refine = iou_ssc.tolist()
                res_dic_refine = {
                    "SC_refine_Precision": precision,
                    "SC_refine_Recall": recall,
                    "SC_refine_IoU": iou,
                    "SSC_refine_mIoU": iou_ssc[1:17].mean(),
                }
                res_dic.update(res_dic_refine)

        class_names = [
            'empty', 'barrier', 'bicycle', 'bus', 'car',
            'construction_vehicle', 'motorcycle', 'pedestrian',
            'traffic_cone', 'trailer', 'truck', 'driveable_surface',
            'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation',
        ]

        for name, iou in zip(class_names, class_ssc_iou):
            res_dic["SSC_{}_IoU".format(name)] = iou
        if 'ssc_results_refine' in results:
            for name, iou in zip(class_names, class_ssc_iou_refine):
                res_dic["SSC_refine_{}_IoU".format(name)] = iou

        for key, val in res_dic.items():
            eval_results['nuScenes_{}'.format(key)] = round(val * 100, 2)

        eval_results['nuScenes_combined_IoU'] = (
            eval_results['nuScenes_SC_IoU'] + eval_results['nuScenes_SSC_mIoU'])

        if logger is not None:
            logger.info('NuScenes SSC Evaluation')
            logger.info(eval_results)

        return eval_results

    def evaluate(self, results, logger=None, **kwargs):
        if results is None:
            if logger:
                logger.info('Skip Evaluation')
            return {}

        if 'evaluation_semantic' in results:
            return self.evaluate_lidarseg(results, logger, **kwargs)
        else:
            return self.evaluate_ssc(results, logger, **kwargs)
