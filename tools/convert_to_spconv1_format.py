#!/usr/bin/env python
"""
Convert checkpoint from spconv 2.x format to spconv 1.x format.
The _load_from_state_dict hook will then convert it back to spconv 2.x.
"""
import torch
from collections import OrderedDict

def main():
    input_path = 'projects/LiCROcc/pre_ckpt/merged_model_distill.pth'
    output_path = 'projects/LiCROcc/pre_ckpt/merged_model_distill_spconv1.pth'
    
    print(f'Loading checkpoint from {input_path}...')
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
    
    # Extract state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        checkpoint = {'state_dict': state_dict}
    
    # Convert spconv 2.x weights to spconv 1.x format
    # spconv 2.x: (out_channels, K, K, K, in_channels)
    # spconv 1.x (MMCV): (D, H, W, in_channels, out_channels)
    converted_count = 0
    for key, value in state_dict.items():
        # Check if this is a spconv weight
        if ('backbone' in key and 'spconv_layers' in key and 
            'weight' in key and value.dim() == 5):
            # Convert from (out, K, K, K, in) to (K, K, K, in, out)
            original_shape = value.shape
            # Permute: (0, 1, 2, 3, 4) -> (1, 2, 3, 4, 0)
            converted = value.permute(1, 2, 3, 4, 0).contiguous()
            state_dict[key] = converted
            converted_count += 1
            if converted_count <= 3:
                print(f'  Converted {key}: {original_shape} -> {converted.shape}')
    
    print(f'\nConverted {converted_count} spconv weights')
    print('Format: spconv 2.x -> MMCV spconv 1.x')
    
    # Save without _metadata so hook will perform conversion
    checkpoint['state_dict'] = state_dict
    if '_metadata' in checkpoint:
        del checkpoint['_metadata']
    
    print(f'\nSaving to {output_path}...')
    torch.save(checkpoint, output_path)
    print('Done!')
    print(f'\nUpdate config to use: load_from = \'{output_path}\'')

if __name__ == '__main__':
    main()

