# Get the covariance and mean of the training data.

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow_datasets as tfds
from tqdm import tqdm
import pickle
from jax import numpy as jnp
from jax import vmap

dataset_builder = tfds.builder('imagenet2012')
dataset_builder.download_and_prepare(download_config=tfds.download.DownloadConfig(
    manual_dir='/user/home/qe22442/work/ImageNet'))


from input_pipeline import preprocess_for_eval, create_split

total_sizes = [100000,10000,1000]
total_sizes = [5000]

for total_size in total_sizes:
    ds = create_split(dataset_builder, 1000, True)
    C = 3
    D = 224
    H = 224
    W = 224
    mu_H = jnp.zeros((C, D))
    cov_H = jnp.zeros((C, D, D))
    mu_W = jnp.zeros((C, D))
    cov_W = jnp.zeros((C, D, D))
    cov_counter = 0
    total_counter = 0
    for t in tqdm(tfds.as_numpy(ds)):
        X = t['image'] / 255
        y = t['label']
        N = X.shape[0]
        x_H = X.transpose(3, 0, 2, 1).reshape(C, N * W, H)
        x_W = X.transpose(3, 0, 1, 2).reshape(C, N * H, W)
        batch_size = x_H.shape[1]
        new_cov_H = vmap(lambda x: x.T @ x)(x_H)
        new_mu_H = x_H.sum(1)
        new_cov_W = vmap(lambda x: x.T @ x)(x_W)
        new_mu_W = x_W.sum(1)
        cov_H = cov_H + new_cov_H
        mu_H = mu_H + (new_mu_H - mu_H * batch_size) / (cov_counter + batch_size)
        cov_W = cov_W + new_cov_W
        mu_W =  mu_W + (new_mu_W - mu_W * batch_size) / (cov_counter + batch_size)
        cov_counter = cov_counter + x_H.shape[1]
        total_counter += N
        if total_counter == total_size:
            break

    cov_dict = {
        'mu_H': mu_H,
        'mu_W': mu_W,
        'cov_H': cov_H,
        'cov_W': cov_W,
        'cov_counter': cov_counter
    }

    with open(os.path.join('./cov_cache', f'normalized_input_cov_{total_size}.pkl'), "wb") as f:
        pickle.dump(cov_dict, f)
