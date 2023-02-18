import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform' # Optional

import tensorflow_datasets as tfds
from tqdm import tqdm
import pickle

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--size', nargs='?', help='<Required> Set flag', required=True, type=int)
args = parser.parse_args()
train_set_size = args.size

dataset_builder = tfds.builder('imagenet2012')
dataset_builder.download_and_prepare(download_config=tfds.download.DownloadConfig(
    manual_dir='/user/home/qe22442/work/ImageNet'))


from input_pipeline import preprocess_for_eval, create_split

ds = create_split(dataset_builder, 1000, True)
img = next(iter(tfds.as_numpy(ds)))['image'][:20]


import haiku as hk
import jax
import jax.numpy as jnp
from PIL import Image

import haikumodels as hm

rng = jax.random.PRNGKey(42)

def _model(images, is_training, sm_mode):
    net = hm.SmResNet50()
    return net(images, is_training, sm_mode)

model = hk.transform_with_state(_model)
params, state = model.init(rng, img, is_training=True, sm_mode=None)

correct = 0
total = 0
for t in tqdm(tfds.as_numpy(ds)):
    X = t['image']
    y = t['label']
    X = hm.resnet.preprocess_input(X)
    preds, state = model.apply(params, state, None, X, is_training=True, sm_mode='acc')
    total += len(X)
    if total >= train_set_size:
        break
    correct += (preds.argmax(-1) == y).sum()

with open(os.path.join('./cov_cache', f"bn_sm_{train_set_size}.pkl"), "wb") as f:
    pickle.dump(state, f)

print(total)
print(correct / total)
