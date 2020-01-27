from keras.layers import Input, Flatten, Dense, Reshape
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.initializers import RandomNormal
import math
import time
import numpy as np
import os
import pickle as pkl

from utils import save_images, show_images


class GAN():
    def __init__(self
                 , input_dim
                 , discriminator_dense
                 , discriminator_learning_rate
                 , generator_dense
                 , generator_learning_rate
                 , optimiser
                 , z_dim
                 ):

        self.name = 'gan'

        self.input_dim = input_dim

        self.discriminator_dense = discriminator_dense
        self.generator_dense = generator_dense
        self.n_layers_generator = len(self.generator_dense)
        self.n_layers_discriminator = len(self.discriminator_dense)

        self.discriminator_learning_rate = discriminator_learning_rate
        self.generator_learning_rate = generator_learning_rate

        self.optimiser = optimiser
        self.z_dim = z_dim

        self.weight_init = RandomNormal(mean=0., stddev=0.02)

        self.d_losses = []
        self.g_losses = []

        self.epoch = 0

        self._build_discriminator()
        self._build_generator()
        self._build_adversarial()

    def _build_discriminator(self):

        ### THE discriminator

        discriminator_input = Input(shape=self.input_dim, name='discriminator_input')
        x = discriminator_input
        x = Flatten()(x)

        for i in range(self.n_layers_discriminator):
            x = Dense(units=self.discriminator_dense[i], activation='relu', kernel_initializer=self.weight_init)(x)

        discriminator_output = Dense(units=1, activation='sigmoid', kernel_initializer=self.weight_init)(x)

        self.discriminator = Model(discriminator_input, discriminator_output)


    def _build_generator(self):

        ### THE generator

        generator_input = Input(shape=(self.z_dim,), name='generator_input')
        x = generator_input

        x = Dense(units=self.generator_dense[0], kernel_initializer=self.weight_init)(x)

        for i in range(self.n_layers_generator):
            x = Dense(units=self.generator_dense[i], activation='relu', kernel_initializer=self.weight_init)(x)

        x = Dense(units=np.prod(self.input_dim), activation='tanh', kernel_initializer=self.weight_init)(x)
        generator_output = Reshape(self.input_dim)(x)

        self.generator = Model(generator_input, generator_output)

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
                , self.discriminator_dense
                , self.discriminator_learning_rate
                , self.generator_dense
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

