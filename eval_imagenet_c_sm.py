# Match scaled input

import os
import pickle
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-l','--level', nargs='+', help='<Required> Set flag', required=True, type=int)
args = parser.parse_args()

import tensorflow_datasets as tfds
from tqdm import tqdm
from get_noise_ds import get_noise_dataset
import haiku as hk
import jax
import jax.numpy as jnp
from PIL import Image
import haikumodels as hm
from input_pipeline import (preprocess_for_eval, create_split,
                            match_channel_joint, match_channel_sep)


def get_dataset(ctype, level, batch_size=1000):
    print(f'Loading {ctype} of level {level}')
    ds = tfds.load(f'imagenet2012_corrupted/{ctype}_{level}',batch_size=batch_size)['validation']
    return ds


CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression', 
    'gaussian_blur'
]

sm_config = {
    'exclude_input': True,
    'input_only': False,
}
INPUT_EPS=1e-3

# ds = get_dataset(ctype, level, 1000)
levels = args.level
print(f'Evaluating on level: {levels}')

for level in levels:
    for noise_type in CORRUPTIONS:
        print(f'Running {noise_type} noise of level {level}')
        ds = get_dataset(noise_type, level, 1000)
        img = next(iter(tfds.as_numpy(ds)))['image'][:20].astype('float32')
        rng = jax.random.PRNGKey(42)
        def _model(images, is_training, sm_mode, **kwargs):
            net = hm.SmResNet50()
            return net(images, is_training, sm_mode, kwargs)

        model = hk.transform_with_state(_model)
        params, state = model.init(rng, img, is_training=True, sm_mode=None, **sm_config)

        with open('./cov_cache/bn_sm.pkl', "rb") as f:
            sm_state = pickle.load(f)
        input_sm_func = match_channel_sep()
        correct = 0
        total = 0
        labels = []
        predss = []
        for t in tqdm(tfds.as_numpy(ds)):
            X = t['image'].astype('float32')
            y = t['label']
            X = input_sm_func(X, INPUT_EPS)
            X = hm.resnet.preprocess_input(X)
            preds, _ = model.apply(params, sm_state, None, X, is_training=True,
                            sm_mode='match', **sm_config)
            total += len(X)
            predss.append(preds)
            labels.append(y)
            correct += (preds.argmax(-1) == y).sum()

        predss = jnp.array(predss)
        labels = jnp.array(labels)

        jnp.save(f'./logs/sm_{noise_type}_{level}_preds.pkl', predss)
        jnp.save(f'./logs/sm_{noise_type}_{level}_label.pkl', labels)

        print(correct / total)
