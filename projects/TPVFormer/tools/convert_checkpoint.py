"""Convert original TPVFormer checkpoint to new format compatible with mmdetection3d."""

import torch
import argparse
from pathlib import Path


def convert_checkpoint(input_path, output_path):
    """Convert checkpoint keys from original to new format.
    
    Args:
        input_path (str): Path to original checkpoint
        output_path (str): Path to save converted checkpoint
    """
    print(f"Loading checkpoint from: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
    
    # mmengine removes 'module.' prefix, so we need to handle both cases
    state_dict = checkpoint if isinstance(checkpoint, dict) and any('module.' in k for k in checkpoint.keys()) else checkpoint
    
    # Create new state dict with converted keys
    new_state_dict = {}
    
    # Key conversion mapping
    key_mappings = {
        'module.img_backbone.': 'backbone.',
        'module.img_neck.': 'neck.',
        'module.tpv_head.': 'tpv_head.',
        'module.tpv_aggregator.': 'tpv_aggregator.',
        'img_backbone.': 'backbone.',
        'img_neck.': 'neck.',
    }
    
    # TPV head specific conversions
    tpv_head_key_mappings = {
        'tpv_head.level_embeds': 'tpv_head.encoder.level_embeds',
        'tpv_head.cams_embeds': 'tpv_head.encoder.cams_embeds',
        'tpv_head.tpv_embedding_hw.weight': 'tpv_head.encoder.tpv_embedding_hw.weight',
        'tpv_head.tpv_embedding_zh.weight': 'tpv_head.encoder.tpv_embedding_zh.weight',
        'tpv_head.tpv_embedding_wz.weight': 'tpv_head.encoder.tpv_embedding_wz.weight',
    }
    
    # Keys to ignore (ref_2d is old format, not needed)
    keys_to_ignore = {
        'tpv_head.tpv_mask_hw',
        'tpv_head.encoder.ref_2d_hw',
        'tpv_head.encoder.ref_2d_zh',
        'tpv_head.encoder.ref_2d_wz',
        'module.tpv_head.tpv_mask_hw',
        'module.tpv_head.encoder.ref_2d_hw',
        'module.tpv_head.encoder.ref_2d_zh',
        'module.tpv_head.encoder.ref_2d_wz',
    }
    
    # Note: ref_3d_* keys should be included (not ignored) as they affect model performance
    
    converted_count = 0
    ignored_count = 0
    
    for key, value in state_dict.items():
        # Check if key should be ignored
        if key in keys_to_ignore:
            print(f"Ignoring: {key}")
            ignored_count += 1
            continue
        
        new_key = key
        
        # Apply prefix mappings
        for old_prefix, new_prefix in key_mappings.items():
            if new_key.startswith(old_prefix):
                new_key = new_key.replace(old_prefix, new_prefix, 1)
                break
        
        # Apply specific TPV head mappings
        if new_key in tpv_head_key_mappings:
            new_key = tpv_head_key_mappings[new_key]
        
        if new_key != key:
            print(f"Converting: {key} -> {new_key}")
            converted_count += 1
        
        new_state_dict[new_key] = value
    
    print(f"\nConversion complete:")
    print(f"  - Total keys: {len(state_dict)}")
    print(f"  - Converted: {converted_count}")
    print(f"  - Ignored: {ignored_count}")
    print(f"  - Unchanged: {len(state_dict) - converted_count - ignored_count}")
    
    # Save converted checkpoint
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(new_state_dict, output_path)
    print(f"\nSaved converted checkpoint to: {output_path}")
    
    # Verify key prefixes
    print("\nKey prefixes in converted checkpoint:")
    prefixes = set()
    for key in new_state_dict.keys():
        prefix = key.split('.')[0] if '.' in key else key
        prefixes.add(prefix)
    for prefix in sorted(prefixes):
        count = sum(1 for k in new_state_dict.keys() if k.startswith(prefix + '.'))
        print(f"  - {prefix}.*: {count} keys")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert TPVFormer checkpoint format')
    parser.add_argument('input', type=str, help='Path to original checkpoint')
    parser.add_argument('output', type=str, help='Path to save converted checkpoint')
    
    args = parser.parse_args()
    
    convert_checkpoint(args.input, args.output)

