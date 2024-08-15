import cv2
import numpy as np
import utils as ut


def show_multliple_blur(image_path):
    new_size = (780, 540)
    new_size = (600, 475)

    max_blur = 10
    level_blur_mean = 0.3
    kernel_gaussian = (13, 13)
    param_bilat = [11, 21, 7]
    median_size = 3

    image = cv2.imread(path)
    image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    window_name = f'Image and different blrurring method for blur level {level_blur_mean * 100} %'

    # Using cv2.blur() method
    ksize = (int(max_blur * level_blur_mean), int(max_blur * level_blur_mean))
    image_mean_blur = cv2.blur(image, ksize)
    image_gaussian_blur = cv2.GaussianBlur(image, kernel_gaussian, 0)
    image_median_blur = cv2.medianBlur(image, median_size)
    image_bilat_blur = cv2.bilateralFilter(image, param_bilat[0], param_bilat[1], param_bilat[2])

    # Formating the images
    numpy_vertical_1 = np.vstack((image, image_mean_blur))
    numpy_vertical_2 = np.vstack((image_gaussian_blur, image_median_blur))
    numpy_vertical_3 = np.vstack((image_bilat_blur, np.zeros_like(image_gaussian_blur)))
    numpy_horizontal_1 = np.hstack((numpy_vertical_1, numpy_vertical_2, numpy_vertical_3))

    cv2.imshow(window_name, numpy_horizontal_1)
    cv2.waitKey(0)

def multiple_median(image_path, resize = True, Save = None, time_stamp = False):
    new_size = (500, 350)
    max_blur = 20
    window_name = f'Mean blurred images with max blur {max_blur}'


    blured = []
    for i in range(1, 10,):
        image = cv2.imread(image_path)
        if resize:
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

        Ksize = (int(max_blur/10*i), int(max_blur/10*i))
        print(Ksize)
        blurred_image = cv2.blur(image, Ksize)
        blured.append(blurred_image)


        if time_stamp:
            image = ut.stamp_image(blurred_image, incr= 2.5)

        if Save is not None:
            name = f'blurred_003_mean_level_{i}'
            path = f'{Save}/{name}.jpg'
            cv2.imwrite(path, blurred_image)
            print(f'Image saved at {Save}')



    vert_1 = np.vstack((blured[0], blured[1], blured[2]))
    vert_2 = np.vstack((blured[3], blured[4], blured[5]))
    vert_3 = np.vstack((blured[6], blured[7], blured[8]))
    hor = np.hstack((vert_1, vert_2, vert_3))

    cv2.imshow(window_name, hor)
    cv2.waitKey(0)

def add_stamp(img_path, save_path=None):
    image = cv2.imread(img_path)
    if save_path is not None:
        ut.stamp_image(image, save= True, save_path= save_path)
    else:
        ut.stamp_image(image, display=True)


if __name__ == "__main__":
    # path
    path = '../01_IMAGES/images_CV/INSP_02_010.jpg'
    #path = '../01_IMAGES/images_CV/test_time_stamp_001.jpg '
    save_path = '../01_IMAGES/Level_images/Mean'

    # Test Functions
    # show_multliple_blur(path)
    #multiple_median(path, resize=False, Save=save_path, time_stamp=True)
    #add_stamp(path, '../01_IMAGES/images_CV/test_time_stamp_001.jpg ')
    p1 = 'C:/Users/Local IT/Downloads/picture4.jpg'
    p2 = '../01_IMAGES/Level_images/Mean/blurred_mean_level_4.jpg'
    im1 = cv2.imread(p1)
    im1 = cv2.resize(im1, (452, 640), interpolation=cv2.INTER_LINEAR)
    im2 = cv2.imread(p2)
    print(im1.shape)
    print(im2.shape)
    cv2.imshow('trest', np.hstack((im1, im2)))
    cv2.waitKey(0)
    arr = im1-im2
    cv2.imshow('trest', arr)
    cv2.waitKey(0)
