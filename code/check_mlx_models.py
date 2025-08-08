#!/usr/bin/env python3
import mlx_lm.models
import os

models_path = mlx_lm.models.__path__[0]
print('MLX-LM models directory:', models_path)
print('Available models:')
for item in os.listdir(models_path):
    if not item.startswith('_') and not item.endswith('.pyc'):
        print('  ', item)