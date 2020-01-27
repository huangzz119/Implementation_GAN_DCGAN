from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Activation, \
    BatchNormalization, LeakyReLU, Dropout, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.initializers import RandomNormal
import math
import time
import numpy as np
import os
import pickle as pkl

from utils import save_images, show_images

class DCGAN():
    def __init__(self
                 , input_dim
                 , discriminator_conv_filters
                 , discriminator_conv_kernel_size
                 , discriminator_conv_strides
                 , discriminator_batch_norm_momentum
                 , discriminator_activation
                 , discriminator_dropout_rate
                 , discriminator_learning_rate
                 , generator_initial_dense_layer_size
                 , generator_upsample
                 , generator_conv_filters
                 , generator_conv_kernel_size
                 , generator_conv_strides
                 , generator_batch_norm_momentum
                 , generator_activation
                 , generator_dropout_rate
                 , generator_learning_rate
                 , optimiser
                 , z_dim
                 ):

        self.name = 'dcgan'

        self.input_dim = input_dim
        self.discriminator_conv_filters = discriminator_conv_filters
        self.discriminator_conv_kernel_size = discriminator_conv_kernel_size
        self.discriminator_conv_strides = discriminator_conv_strides
        self.discriminator_batch_norm_momentum = discriminator_batch_norm_momentum
        self.discriminator_activation = discriminator_activation
        self.discriminator_dropout_rate = discriminator_dropout_rate
        self.discriminator_learning_rate = discriminator_learning_rate

        self.generator_initial_dense_layer_size = generator_initial_dense_layer_size
        self.generator_upsample = generator_upsample
        self.generator_conv_filters = generator_conv_filters
        self.generator_conv_kernel_size = generator_conv_kernel_size
        self.generator_conv_strides = generator_conv_strides
        self.generator_batch_norm_momentum = generator_batch_norm_momentum
        self.generator_activation = generator_activation
        self.generator_dropout_rate = generator_dropout_rate
        self.generator_learning_rate = generator_learning_rate

        self.optimiser = optimiser
        self.z_dim = z_dim

        self.n_layers_discriminator = len(discriminator_conv_filters)
        self.n_layers_generator = len(generator_conv_filters)

        self.weight_init = RandomNormal(mean=0., stddev=0.02)

        self.d_losses = []
        self.g_losses = []

        self.epoch = 0

        self._build_discriminator()
        self._build_generator()

        self._build_adversarial()

    def get_activation(self, activation):
        """
        choose the activative function
        :param activation: the name of activation
        :return:
        """
        if activation == 'leaky_relu':
            layer = LeakyReLU(alpha=0.2)
        else:
            layer = Activation(activation)
        return layer

    def _build_discriminator(self):

        ### THE discriminator

        discriminator_input = Input(shape=self.input_dim, name='discriminator_input')
        x = discriminator_input

        for i in range(self.n_layers_discriminator):

            x = Conv2D(
                filters=self.discriminator_conv_filters[i]
                , kernel_size=self.discriminator_conv_kernel_size[i]
                , strides=self.discriminator_conv_strides[i]
                , padding='same'
                , name='discriminator_conv_' + str(i)
                , kernel_initializer=self.weight_init
            )(x)

            if self.discriminator_batch_norm_momentum and i > 0:
                x = BatchNormalization(momentum=self.discriminator_batch_norm_momentum)(x)

            x = self.get_activation(self.discriminator_activation)(x)

            if self.discriminator_dropout_rate:
                x = Dropout(rate=self.discriminator_dropout_rate)(x)

        x = Flatten()(x)
        discriminator_output = Dense(1, activation='sigmoid', kernel_initializer=self.weight_init)(x)
        self.discriminator = Model(discriminator_input, discriminator_output)

    def _build_generator(self):

        ### THE generator

        generator_input = Input(shape=(self.z_dim,), name='generator_input')

        x = generator_input

        x = Dense(np.prod(self.generator_initial_dense_layer_size), kernel_initializer=self.weight_init)(x)

        if self.generator_batch_norm_momentum:
            x = BatchNormalization(momentum=self.generator_batch_norm_momentum)(x)

        x = self.get_activation(self.generator_activation)(x)

        x = Reshape(self.generator_initial_dense_layer_size)(x)

        if self.generator_dropout_rate:
            x = Dropout(rate=self.generator_dropout_rate)(x)

        for i in range(self.n_layers_generator):

            if self.generator_upsample[i] == 2:
                x = UpSampling2D()(x)
                x = Conv2D(
                    filters=self.generator_conv_filters[i]
                    , kernel_size=self.generator_conv_kernel_size[i]
                    , padding='same'
                    , name='generator_conv_' + str(i)
                    , kernel_initializer=self.weight_init
                )(x)
            else:

                x = Conv2DTranspose(
                    filters=self.generator_conv_filters[i]
                    , kernel_size=self.generator_conv_kernel_size[i]
                    , padding='same'
                    , strides=self.generator_conv_strides[i]
                    , name='generator_conv_' + str(i)
                    , kernel_initializer=self.weight_init
                )(x)

            if i < self.n_layers_generator - 1:

                if self.generator_batch_norm_momentum:
                    x = BatchNormalization(momentum=self.generator_batch_norm_momentum)(x)

                x = self.get_activation(self.generator_activation)(x)


            else:
                # tansform the output to the range [-1, 1] to match the original image domain
                x = Activation('tanh')(x)

        generator_output = x
        self.generator = Model(generator_input, generator_output)

    def get_opti(self, lr):
        """
        choose the optimization function
        :param lr: the name of optimiser
        """
        if self.optimiser == 'adam':
            opti = Adam(lr=lr, beta_1=0.5)
        elif self.optimiser == 'rmsprop':
            opti = RMSprop(lr=lr)
        else:
            opti = Adam(lr=lr)
        return opti

    def set_trainable(self, m, val):
        """
        set the training parameters
        :param m: the model
        :param val: false or true
        """
        m.trainable = val
        for l in m.layers:
            l.trainable = val

    def _build_adversarial(self):

        ### COMPILE DISCRIMINATOR

        self.discriminator.compile(
            optimizer=self.get_opti(self.discriminator_learning_rate)
            , loss='binary_crossentropy'
            , metrics=['accuracy']
        )

        ### COMPILE THE FULL GAN

        self.set_trainable(self.discriminator, False)

        model_input = Input(shape=(self.z_dim,), name='model_input')
        model_output = self.discriminator(self.generator(model_input))
        self.model = Model(model_input, model_output)

        self.model.compile(optimizer=self.get_opti(self.generator_learning_rate), loss='binary_crossentropy',
                           metrics=['accuracy'])

        self.set_trainable(self.discriminator, True)

    def train_discriminator(self, x_train, batch_size, using_generator):
        """
        this function is used to train discriminator
        :param x_train: the input image
        :param batch_size: the batch size
        :param using_generator: whether to use generator or not
        :return: the loss of discriminator,
                the loss of discriminator for real image,
                the loss of discriminator of fake image,
                the accuracy of discriminator,
                the accuracy of discriminator for real image,
                the accuracy of discriminator for fake image.
        """

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        if using_generator:
            true_imgs = next(x_train)[0]
            if true_imgs.shape[0] != batch_size:
                true_imgs = next(x_train)[0]
        else:
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            true_imgs = x_train[idx]

        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        d_loss_real, d_acc_real = self.discriminator.train_on_batch(true_imgs, valid)
        d_loss_fake, d_acc_fake = self.discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        d_acc = 0.5 * (d_acc_real + d_acc_fake)

        return [d_loss, d_loss_real, d_loss_fake, d_acc, d_acc_real, d_acc_fake]

    def train_generator(self, batch_size):
        """
        this function is used to train generator
        :return: the training model
        """
        valid = np.ones((batch_size, 1))
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        return self.model.train_on_batch(noise, valid)

    def train(self, x_train, batch_size, epochs, run_folder
              , print_every_n_batches=50
              , using_generator=False):
        """
        this function is used to train the dataset
        :param x_train: the input image
        :param batch_size: the batch size
        :param epochs: the total epochs
        :param run_folder: the folder to save image
        :param print_every_n_batches: the batches number to print
        :param using_generator: whether use generator or not
        """

        start_time = time.time()
        init_epoch = self.epoch

        for epoch in range(self.epoch, self.epoch + epochs):

            d = self.train_discriminator(x_train, batch_size, using_generator)
            g = self.train_generator(batch_size)

            self.d_losses.append(d)
            self.g_losses.append(g)

            if epoch % print_every_n_batches == 0:
                print("Epoch: %d ", epoch)
                print("Discriminator loss: (%.3f)(Real %.3f, Fake %.3f)", d[0], d[1], d[2])
                print("Discriminator accuracy: (%.3f)(Real %.3f,Fake %.3f)", d[3], d[4], d[5])
                print("Generator loss: %.3f,  Generator accuracy: %.3f]", g[0], g[1])

                image_path = os.path.join(run_folder, "images/sample_%d.png" % self.epoch)
                self.sample_images(batch_size, image_path)
                self.model.save_weights(os.path.join(run_folder, 'weights/weights-%d.h5' % (epoch)))
                self.model.save_weights(os.path.join(run_folder, 'weights/weights.h5'))
                self.save_model(run_folder)

                # Print some performance statistics.
                end_time = time.time()
                current_epoch = epoch - init_epoch
                time_taken = end_time - start_time
                print("Current epoch: %i,  time since start: %.1f sec" % (current_epoch, time_taken))

                show_images(image_path)
                # Generate and show some predictions.
            self.epoch += 1

    def sample_images(self, batch_size, run_folder):
        """
        save the sample images during the training
        :param run_folder: the folder to save image
        """
        image_frame_dim = int(math.ceil(batch_size ** .5))
        noise = np.random.normal(0, 1, (image_frame_dim * image_frame_dim, self.z_dim))
        gen_imgs = self.generator.predict(noise)
        save_images(gen_imgs, [image_frame_dim, image_frame_dim], run_folder)



    def save(self, folder):
        """
        this function is used to save all the model information
        :param folder: the folder to save data
        """
        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pkl.dump([
                self.input_dim
                , self.discriminator_conv_filters
                , self.discriminator_conv_kernel_size
                , self.discriminator_conv_strides
                , self.discriminator_batch_norm_momentum
                , self.discriminator_activation
                , self.discriminator_dropout_rate
                , self.discriminator_learning_rate
                , self.generator_initial_dense_layer_size
                , self.generator_upsample
                , self.generator_conv_filters
                , self.generator_conv_kernel_size
                , self.generator_conv_strides
                , self.generator_batch_norm_momentum
                , self.generator_activation
                , self.generator_dropout_rate
                , self.generator_learning_rate
                , self.optimiser
                , self.z_dim
            ], f)

    def save_model(self, run_folder):
        """
        this function is used to save model training parameters
        """
        self.model.save(os.path.join(run_folder, 'model.h5'))
        self.discriminator.save(os.path.join(run_folder, 'discriminator.h5'))
        self.generator.save(os.path.join(run_folder, 'generator.h5'))
        pkl.dump(self, open(os.path.join(run_folder, "obj.pkl"), "wb"))

    def load_weights(self, filepath):
        """
        this function is used to load former weights
        """
        self.model.load_weights(filepath)




