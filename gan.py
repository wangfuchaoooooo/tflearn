""" GAN Example
Use a generative adversarial network (GAN) to generate digit images from a
noise distribution.
References:
    - Generative adversarial nets. I Goodfellow, J Pouget-Abadie, M Mirza,
    B Xu, D Warde-Farley, S Ozair, Y. Bengio. Advances in neural information
    processing systems, 2672-2680.
Links:
    - [GAN Paper](https://arxiv.org/pdf/1406.2661.pdf).
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tflearn

import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data('data/')

image_dim = 784  # 28 X 28 pixels
z_dim = 200  # Noise data points
total_samples = len(X)


# Generator
def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        x = tflearn.fully_connected(x, 256, activation='relu')
        x = tflearn.fully_connected(x, image_dim, activation='sigmoid')
        return x


# Discriminator
def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        x = tflearn.fully_connected(x, 256, activation='relu')
        x = tflearn.fully_connected(x, 1, activation='sigmoid')
        return x


# Build Networks
gen_input = tflearn.input_data(shape=[None, z_dim], name='input_noise')
disc_input = tflearn.input_data(shape=[None, 784], name='disc_input')

gen_sample = generator(gen_input)
disc_real = discriminator(disc_input)
disc_fake = discriminator(gen_sample, reuse=True)

# Define Loss
disc_real = tf.clip_by_value(disc_real, 1e-10, 1e100)
disc_fake = tf.clip_by_value(disc_fake, 1e-10, 1e100)   # 防止 loss 出现 NAN 结果
disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))
gen_loss = -tf.reduce_mean(tf.log(disc_fake))

# Build training Ops for both Generator and Discriminator

# Each network optimization should only its own variable, thus we need
# to retrieve each network variable(with get_layer_variables_by_scop) and set
# 'placeholder=None' because we do not need to feed any target

gen_vars = tflearn.get_layer_variables_by_scope('Generator')
gen_model = tflearn.regression(gen_sample, placeholder=None, optimizer='adam',
                               loss=gen_loss, trainable_vars=gen_vars,
                               batch_size=64, name='target_gen', op_name='GEN')

disc_vars = tflearn.get_layer_variables_by_scope('Discriminator')
disc_model = tflearn.regression(disc_real, placeholder=None, optimizer='adam',
                                loss=disc_loss, trainable_vars=disc_vars,
                                batch_size=64, name='target_disc', op_name='DISC')

# Define GAN model, that output the generated images
gan = tflearn.DNN(gen_model)

# Training
# Generator noise to feed to the generator
z = np.random.uniform(-1., 1., size=[total_samples, z_dim])

# Start training, feed both noise and real images.
gan.fit(X_inputs={gen_input: z, disc_input: X},
        Y_targets=None, n_epoch=100)

f, a = plt.subplots(2, 10, figsize=(10, 4))
for i in range(10):
    for j in range(2):
        # Noise Input
        z = np.random.uniform(-1., 1., size=[1, z_dim])
        temp = [[ii, ii, ii] for ii in list(gan.predict([z])[0])]
        a[j][i].imshow(np.reshape(temp, (28, 28, 3)))
f.show()
plt.draw()
plt.waitforbuttonpress()