from __future__ import division
import cv2
import numpy as np
import os
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from os import listdir
from numpy import asarray

# method 1
def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64):
    """
    this function is to change the size of image.
    :param image_path: the path of image
    :param input_height: the height of image before resize
    :param input_width: the width of image before resize
    :param resize_height: the resized height of image
    :param resize_width: the resized width of image
    :return: the transform function
    """
    try:
        img_bgr = cv2.imread(image_path)     # OpenCV reads image
        img_rgb = img_bgr[..., ::-1]  # change BGR to RGB
        image = img_rgb.astype(np.float)
        return transform(image, input_height, input_width,
                         resize_height, resize_width)
    except:
        return False

def transform(image, input_height, input_width, resize_height=64, resize_width=64):
    """
    firstly, the original image is changed as "input image" around the center
    secondly, the "input image" is resized as output
    :param image:
    :return: the array of the image
    """
    h, w = image.shape[:2]
    j = int(round((h - input_height) / 2.))
    i = int(round((w - input_width) / 2.))
    im = Image.fromarray(image[j:j + input_height, i:i + input_width].astype(np.uint8))
    return (np.array(im.resize([resize_height, resize_width], Image.BILINEAR)) / 127.5 - 1.).tolist()


# method 2
def load_images(path, size=(64, 64)):
    """
    this function is to load all images in a directory into memory
    :param path: the path of the original image
    :param size: the target size of image
    :return: the array of all images
    """
    data_list = list()
    for filename in listdir(path):
        try:
            pixels = load_img(path + filename, target_size=size)   # load and resize the image
            pixels = img_to_array(pixels)   # convert to numpy array
            data_list.append(pixels)
        except:
            pass
    return asarray(data_list)

if __name__ == '__main__':

    # method 1
    IMAGE_FOLDER = os.path.join(os.getcwd(), 'out_data')
    PROCESS_FOLDER = os.path.join(os.getcwd(), 'process_data/')
    dataset_name = ["fruit/", "graffiti/", "metal/"]
    process_image_name = ["fruit_image.npy", "graffiti_image.npy", "metal_image.npy" ]

    for i in np.arange(len(dataset_name)):
        original_image = os.path.join(IMAGE_FOLDER, dataset_name[i])
        exprocess_image = []
        for image in os.listdir(original_image):
            ex_image = get_image(original_image + image, input_height =128, input_width=128,
                                 resize_height=64,resize_width=64)
            if ex_image != False:
                exprocess_image.append(ex_image)
        process_image = np.array(exprocess_image).astype(np.float32)
        np.save(PROCESS_FOLDER+ process_image_name[i], process_image)

    #load the dataset
    training_data = np.load(PROCESS_FOLDER+'metal_image.npy')


    # method 2
    path = os.getcwd() + '/out_data/'
    dataA = load_images(path + "fruit/")
    dataB = load_images(path + "graffiti/")
    dataC = load_images(path + "metal/")

    filename = 'out_data.npz'
    np.savez_compressed(filename, dataA, dataB, dataC)
    print('Saved dataset: ', filename)

    # load the dataset
    data = np.load('out_data.npz')
    dataA, dataB, dataC = data['arr_0'], data['arr_1'], data['arr_2']
    print('Loaded: ', dataA.shape, dataB.shape, dataC.shape)







