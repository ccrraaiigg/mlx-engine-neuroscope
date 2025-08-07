#!/usr/bin/env python3
"""
Download the 4-bit MLX model for gpt-oss-20b
"""

from huggingface_hub import snapshot_download
import os

print('Downloading nightmedia/gpt-oss-20b-q4-hi-mlx...')
try:
    snapshot_download(
        repo_id='nightmedia/gpt-oss-20b-q4-hi-mlx',
        local_dir='./models/nightmedia/gpt-oss-20b-q4-hi-mlx',
        local_dir_use_symlinks=False
    )
    print('Download completed successfully!')
except Exception as e:
    print(f'Download failed: {e}')