import tensorflow_datasets as tfds

def get_noise_dataset(noise_type, level):
    print(f'Using noise {noise_type} of level {level}')
    ds = tfds.ImageFolder(
        f'/user/home/qe22442/work/imagenet/imagenet-c/{noise_type}_noise').as_dataset(
    split=str(level),
    batch_size=1000
    )
    return ds