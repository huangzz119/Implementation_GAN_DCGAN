"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division

import os
import sys
fileDir = os.getcwd()
sys.path.append(fileDir)

import math
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
from PIL import Image
import matplotlib.pyplot as plt
import imageio
from matplotlib import pyplot

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings
warnings.filterwarnings('ignore')


def init_result(MODE = "build"):
    """
    this function is used to build folders
    :param MODE:
    :return:
    """
    if MODE == "build":
        OBJECTS = ["fruit", "graffiti","metal"]
        MODEL = ["GAN","DCGAN"]
        for md in MODEL:
            RUN_FOLDER1 = os.path.join(fileDir, 'Result/{}'.format(md))
            if not os.path.isdir(RUN_FOLDER1):
                os.makedirs(RUN_FOLDER1)
            for ob in OBJECTS:
                PATH_MODEL = os.path.join(RUN_FOLDER1, ob)
                if not os.path.isdir(PATH_MODEL):
                    os.mkdir(PATH_MODEL)
                    os.mkdir(os.path.join(PATH_MODEL, 'viz'))
                    os.mkdir(os.path.join(PATH_MODEL, 'images'))
                    os.mkdir(os.path.join(PATH_MODEL, 'weights'))


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')

def save_images(images, size, image_path):
    transform_images = (images + 1.) / 2.
    # constrain between 0 and 1
    transform_images = np.clip(transform_images, 0, 1)
    image =  np.squeeze(merge(transform_images, size))
    return imageio.imwrite(image_path, image)

def show_images(image_path):
    plt.axis('off')
    plt.imshow(Image.open(image_path))
    plt.show()


def visualize(sess, dcgan, config, option, sample_dir='samples'):
    image_frame_dim = int(math.ceil(config.batch_size ** .5))
    if option == 0:
        z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
        save_images(samples, [image_frame_dim, image_frame_dim],
                    os.path.join(sample_dir, 'test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime())))
    elif option == 1:
        values = np.arange(0, 1, 1. / config.batch_size)
        for idx in xrange(dcgan.z_dim):
            print(" [*] %d" % idx)
            z_sample = np.random.uniform(-1, 1, size=(config.batch_size, dcgan.z_dim))
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            if config.dataset == "mnist":
                y = np.random.choice(10, config.batch_size)
                y_one_hot = np.zeros((config.batch_size, 10))
                y_one_hot[np.arange(config.batch_size), y] = 1

                samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
            else:
                samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

            save_images(samples, [image_frame_dim, image_frame_dim],
                        os.path.join(sample_dir, 'test_arange_%s.png' % (idx)))


def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w


def picture(data, n_samples, PATH, name):
    for j in range(n_samples):
        for i in range(n_samples):
            pyplot.subplot(n_samples, n_samples, 1 + i + j*n_samples)
            pyplot.axis('off')
            pyplot.imshow(data[i + 10 + j*n_samples].astype('uint8'))
    pyplot.savefig( PATH + name + ".png")

