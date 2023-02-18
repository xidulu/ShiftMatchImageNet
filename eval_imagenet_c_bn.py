import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow_datasets as tfds
from tqdm import tqdm
from get_noise_ds import get_noise_dataset
from input_pipeline import preprocess_for_eval, create_split
import haiku as hk
import jax
import jax.numpy as jnp
from PIL import Image

import haikumodels as hm


def get_dataset(ctype, level, batch_size=200):
    print(f'Loading {ctype} of level {level}')
    ds = tfds.load(f'imagenet2012_corrupted/{ctype}_{level}',batch_size=batch_size)['validation']
    return ds

ctype='gaussian_noise'
level=3

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression', 'gaussian_blur'
]


# ds = get_dataset(ctype, level, 1000)
for level in [1,2,3,4,5]:
    for noise_type in CORRUPTIONS:
        print(f'Running {noise_type} noise of level {level}')
        ds = get_dataset(noise_type, level)
        img = next(iter(tfds.as_numpy(ds)))['image'][:20].astype('float32')
        rng = jax.random.PRNGKey(42)

        def _model(images, is_training):
            net = hm.ResNet50()
            return net(images, is_training)


        model = hk.transform_with_state(_model)

        params, state = model.init(rng, img, is_training=True)
        img = hm.resnet.preprocess_input(img)
        preds, _ = model.apply(params, state, None, img, is_training=False)

        correct = 0
        total = 0
        labels = []
        predss = []
        for t in tqdm(tfds.as_numpy(ds)):
            X = t['image']
            y = t['label']
            X = hm.resnet.preprocess_input(X)
            # print(X.shape)
            preds, _ = model.apply(params, state, None, X, is_training=True)
            total += len(X)
            predss.append(preds)
            labels.append(y)
            correct += (preds.argmax(-1) == y).sum()
        jnp.save(f'./logs/bn_{noise_type}_{level}_preds.pkl', predss)
        jnp.save(f'./logs/bn_{noise_type}_{level}_label.pkl', labels)
        print(correct / total)
