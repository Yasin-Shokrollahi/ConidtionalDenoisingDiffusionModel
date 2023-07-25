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

# architecture
embedding_dims = 64
embedding_max_frequency = 1000.0
widths = [32, 64, 128, 256, 512]
block_depth = 2

# optimization
batch_size = 4
ema = 0.999
learning_rate = 1e-4
weight_decay = 1e-8
```

