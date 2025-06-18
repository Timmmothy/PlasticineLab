from setuptools import setup

install_requires = ['scipy', 'numpy', 'torch', 'opencv-python', 'tqdm', 'taichi', 'gymnasium', 'tensorboard', 'yacs', 'matplotlib',
                    'descartes', 'shapely', 'natsort', 'torchvision', 'einops', 'alphashape', 'tensorboardX', 'open3d']

## FOR PPO
# UNCOMMENT
# install_requires.extend(['gym', 'tensorflow'])
# RUN
# pip install baselines@git+https://github.com/openai/baselines@ea25b9e8

setup(name='plb',
      version='0.0.1',
      install_requires=install_requires,
      py_modules=['plb'],
      )
