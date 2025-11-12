"""
Layer-by-Layer Comparison Utility
각 모듈의 입력/출력을 상세히 기록하여 원본과 비교
"""
import json
import torch
import numpy as np
from pathlib import Path


class LayerDebugger:
    """레이어별 디버그 정보를 수집하고 저장하는 클래스"""
    
    def __init__(self, save_dir="debug_outputs"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.stats = {}
        self.sample_idx = 0
        
    def set_sample_idx(self, idx):
        """현재 처리 중인 샘플 인덱스 설정"""
        self.sample_idx = idx
        
    def compute_tensor_stats(self, tensor, name):
        """텐서의 상세 통계 계산"""
        if tensor is None:
            return {"type": "None"}
            
        if not isinstance(tensor, torch.Tensor):
            return {"type": str(type(tensor))}
        
        # Move to CPU for computation
        t = tensor.detach().cpu().float()
        
        stats = {
            "shape": list(t.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "mean": float(t.mean()),
            "std": float(t.std()),
            "min": float(t.min()),
            "max": float(t.max()),
            "abs_mean": float(t.abs().mean()),
            "abs_max": float(t.abs().max()),
        }
        
        # Non-zero ratio
        stats["non_zero_ratio"] = float((t != 0).float().mean())
        
        # Negative ratio
        stats["negative_ratio"] = float((t < 0).float().mean())
        
        # Percentiles
        flat = t.flatten()
        if flat.numel() > 0:
            percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]
            percentile_vals = np.percentile(flat.numpy(), percentiles)
            stats["percentiles"] = {f"p{p}": float(v) for p, v in zip(percentiles, percentile_vals)}
        
        # Histogram (10 bins)
        hist, bin_edges = np.histogram(flat.numpy(), bins=10)
        stats["histogram"] = {
            "counts": hist.tolist(),
            "bin_edges": bin_edges.tolist()
        }
        
        return stats
    
    def log_module_io(self, module_name, input_tensors=None, output_tensors=None, 
                      extra_info=None, save=True):
        """모듈의 입출력 정보 기록
        
        Args:
            module_name: 모듈 이름 (예: "image_backbone", "depthnet", "voxel_pooling")
            input_tensors: 입력 텐서 dict
            output_tensors: 출력 텐서 dict
            extra_info: 추가 정보 dict
            save: 즉시 저장할지 여부
        """
        entry = {
            "sample_idx": self.sample_idx,
            "module": module_name,
            "inputs": {},
            "outputs": {},
            "extra": extra_info or {}
        }
        
        # Process inputs
        if input_tensors:
            for key, tensor in input_tensors.items():
                if isinstance(tensor, (list, tuple)):
                    entry["inputs"][key] = [
                        self.compute_tensor_stats(t, f"{key}_{i}") 
                        for i, t in enumerate(tensor)
                    ]
                else:
                    entry["inputs"][key] = self.compute_tensor_stats(tensor, key)
        
        # Process outputs
        if output_tensors:
            for key, tensor in output_tensors.items():
                if isinstance(tensor, (list, tuple)):
                    entry["outputs"][key] = [
                        self.compute_tensor_stats(t, f"{key}_{i}") 
                        for i, t in enumerate(tensor)
                    ]
                else:
                    entry["outputs"][key] = self.compute_tensor_stats(tensor, key)
        
        # Store in memory
        if module_name not in self.stats:
            self.stats[module_name] = []
        self.stats[module_name].append(entry)
        
        # Save to file if requested
        if save:
            self.save_stats(module_name)
        
        return entry
    
    def log_weights(self, module_name, model, save=True):
        """모델의 weight 통계 기록"""
        weights_stats = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                weights_stats[name] = self.compute_tensor_stats(param.data, name)
        
        entry = {
            "sample_idx": self.sample_idx,
            "module": module_name,
            "weights": weights_stats
        }
        
        # Store in memory
        weight_key = f"{module_name}_weights"
        if weight_key not in self.stats:
            self.stats[weight_key] = []
        self.stats[weight_key].append(entry)
        
        # Save to file if requested
        if save:
            self.save_stats(weight_key)
        
        return entry
    
    def save_stats(self, module_name=None):
        """통계를 JSON 파일로 저장
        
        Args:
            module_name: 특정 모듈만 저장 (None이면 전체 저장)
        """
        if module_name:
            # Save specific module
            if module_name in self.stats:
                filepath = self.save_dir / f"{module_name}_sample{self.sample_idx}.json"
                with open(filepath, 'w') as f:
                    json.dump(self.stats[module_name], f, indent=2)
        else:
            # Save all
            for key, data in self.stats.items():
                filepath = self.save_dir / f"{key}_sample{self.sample_idx}.json"
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
    
    def compare_with_reference(self, module_name, reference_file):
        """원본 모델의 출력과 비교
        
        Args:
            module_name: 비교할 모듈 이름
            reference_file: 원본 출력이 저장된 JSON 파일 경로
        
        Returns:
            비교 결과 dict
        """
        # Load reference
        with open(reference_file, 'r') as f:
            reference = json.load(f)
        
        # Get current stats
        if module_name not in self.stats or len(self.stats[module_name]) == 0:
            return {"error": "No current stats available"}
        
        current = self.stats[module_name][-1]
        
        # Compare
        comparison = {
            "module": module_name,
            "differences": {}
        }
        
        # Compare outputs
        for key in current.get("outputs", {}).keys():
            if key in reference.get("outputs", {}):
                curr_stats = current["outputs"][key]
                ref_stats = reference["outputs"][key]
                
                if isinstance(curr_stats, dict) and isinstance(ref_stats, dict):
                    diff = {
                        "mean_diff": curr_stats.get("mean", 0) - ref_stats.get("mean", 0),
                        "std_diff": curr_stats.get("std", 0) - ref_stats.get("std", 0),
                        "mean_ratio": curr_stats.get("mean", 0) / (ref_stats.get("mean", 0) + 1e-10),
                        "max_diff": curr_stats.get("max", 0) - ref_stats.get("max", 0),
                        "min_diff": curr_stats.get("min", 0) - ref_stats.get("min", 0),
                    }
                    comparison["differences"][key] = diff
        
        return comparison
    
    def print_summary(self, module_name):
        """모듈의 통계 요약 출력"""
        if module_name not in self.stats or len(self.stats[module_name]) == 0:
            print(f"[{module_name}] No stats available")
            return
        
        entry = self.stats[module_name][-1]
        
        print(f"\n{'='*80}")
        print(f"[{module_name.upper()}] Summary - Sample {entry['sample_idx']}")
        print(f"{'='*80}")
        
        # Print inputs
        if entry.get("inputs"):
            print("\n📥 INPUTS:")
            for key, stats in entry["inputs"].items():
                if isinstance(stats, list):
                    for i, s in enumerate(stats):
                        print(f"  [{key}_{i}] Shape: {s.get('shape')}, Mean: {s.get('mean', 0):.6f}, Std: {s.get('std', 0):.6f}")
                else:
                    print(f"  [{key}] Shape: {stats.get('shape')}, Mean: {stats.get('mean', 0):.6f}, Std: {stats.get('std', 0):.6f}")
        
        # Print outputs
        if entry.get("outputs"):
            print("\n📤 OUTPUTS:")
            for key, stats in entry["outputs"].items():
                if isinstance(stats, list):
                    for i, s in enumerate(stats):
                        print(f"  [{key}_{i}] Shape: {s.get('shape')}, Mean: {s.get('mean', 0):.6f}, Std: {s.get('std', 0):.6f}")
                else:
                    print(f"  [{key}] Shape: {stats.get('shape')}, Mean: {stats.get('mean', 0):.6f}, Std: {stats.get('std', 0):.6f}")
                    print(f"         Range: [{stats.get('min', 0):.6f}, {stats.get('max', 0):.6f}]")
                    print(f"         Non-zero: {stats.get('non_zero_ratio', 0)*100:.2f}%, Negative: {stats.get('negative_ratio', 0)*100:.2f}%")
        
        # Print extra info
        if entry.get("extra"):
            print("\n📋 EXTRA INFO:")
            for key, value in entry["extra"].items():
                print(f"  {key}: {value}")
        
        print(f"{'='*80}\n")


# Global debugger instance
global_debugger = None

def get_debugger(save_dir="debug_outputs"):
    """전역 디버거 인스턴스 가져오기"""
    global global_debugger
    if global_debugger is None:
        global_debugger = LayerDebugger(save_dir=save_dir)
    return global_debugger


def log_layer(module_name, inputs=None, outputs=None, extra=None, print_summary=False):
    """레이어 정보 로깅 (간편 함수)"""
    debugger = get_debugger()
    debugger.log_module_io(module_name, inputs, outputs, extra, save=True)
    
    if print_summary:
        debugger.print_summary(module_name)


def log_model_weights(module_name, model):
    """모델 가중치 로깅 (간편 함수)"""
    debugger = get_debugger()
    debugger.log_weights(module_name, model, save=True)

