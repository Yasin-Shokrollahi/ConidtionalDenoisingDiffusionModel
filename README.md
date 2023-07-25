# Conidtional Denoising Diffusion Model for generating Calcified 2D Artery
This Python script uses TensorFlow and Keras to process and train a neural network model on a dataset of images and corresponding stress maps. The script includes functions for image preprocessing and dataset preparation. It then retrieves image files from specified directories, prepares the dataset, and splits it into training and validation sets.
   1. Kindly download the Artery file that includes two folders of images: Input and Stress Map.
   2. Proceed to execute the code in the Jupyter notebook, ensuring that you're using a GPU for computation.
   3. You're welcome to adjust the hyperparameters in the code as per your requirements.
Here is some Python code:

![My Image](./ezgif-3-45cc3d4bfd.gif)





## Set up:

```python
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import layers
import glob
from tensorflow.keras import regularizers
```


## Hyperparameters
You can change all these hyperparameters based on your projects
```python
dataset_repetitions = 5
num_epochs = 1000  # train for at least 50 epochs for good results
image_size = 128
# KID = Kernel Inception Distance, see related section
kid_image_size = 75
kid_diffusion_steps = 5
plot_diffusion_steps = 20
# sampling
min_signal_rate = 0.02
max_signal_rate = 0.95
embedding_dims = 64
embedding_max_frequency = 1000.0
widths = [32, 64, 128, 256, 512]# architecture
block_depth = 2 # optimization
batch_size = 4
ema = 0.999
learning_rate = 1e-4
weight_decay = 1e-8
```
## Import Local dataset of images containg 1500 2d images with shape of 256*256
make sure that change the address to your local adress
```python

def preprocess_image(image_path, stress_map_path):
    # Process original image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]

    # Process stress map
    stress_map = tf.io.read_file(stress_map_path)
    stress_map = tf.image.decode_png(stress_map, channels=3)
    stress_map = tf.cast(stress_map, tf.float32) / 255.0  # Normalize to [0,1]

    return image, stress_map



def prepare_dataset(image_paths, stress_map_paths):
    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    stress_map_dataset = tf.data.Dataset.from_tensor_slices(stress_map_paths)
    dataset = tf.data.Dataset.zip((image_dataset, stress_map_dataset))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.repeat(dataset_repetitions)
    dataset = dataset.shuffle(10 * batch_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset



# Load your local image dataset
image_paths = glob.glob("D:/Diffusion/Artery/Input_Resized/*.png")
stress_map_paths = glob.glob("D:/Diffusion/Artery/Stress_map_Resized/*.png")

# Sort the paths to ensure they are in the same order
image_paths.sort()
stress_map_paths.sort()

train_dataset = prepare_dataset(image_paths, stress_map_paths)

# Split the dataset for training and validation
num_samples = len(image_paths)
train_size = int(0.8 * num_samples)
val_size = num_samples - train_size

train_dataset = train_dataset.take(train_size)
val_dataset = train_dataset.skip(train_size)




```
