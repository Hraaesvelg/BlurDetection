from imutils import paths
import argparse
import cv2
import os
import time
import datetime

import sys
import argparse
import logging
import pathlib
import json
import numpy as np

def stamp_image(image, size = (780,540), display = False, save = False, save_path = 0, incr = 2.5):
    image = cv2.resize(image, size,interpolation=cv2.INTER_LINEAR)
    window_name = 'Added time stamp'
    time_date = datetime.datetime.now()
    time_date = time_date.strftime("%x") + ' ' + time_date.strftime("%X")

    # Using cv2.putText() method
    image = cv2.putText(image, time_date, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65*incr,
                        (0, 0, 0), 2*int(1*incr), cv2.LINE_AA, False)
    image = cv2.putText(image, time_date, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65*incr,
                        (0, 165, 255), int(1*incr), cv2.LINE_AA, False)

    # Displaying the image
    if display:
        print(f'Image displayed with timestamp: {time_date}')
        cv2.imshow(window_name, image)
        cv2.waitKey(0)

    # Saving the image
    if save:
        cv2.imwrite(save_path, image)
        print(f'Image saved at {save_path}')

    return image

def stamp_directory(path, save_path, display=False, size = (780,540)):
    for image_path in path:
        print(image_path)
        image = cv2.imread(str(image_path))
        if size is not None:
            image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

        window_name = 'Added time stamp'
        time_date = datetime.datetime.now()
        time_date = time_date.strftime("%x") + ' ' + time_date.strftime("%X")

        # Using cv2.putText() method
        image = cv2.putText(image, time_date, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 0, 0), 2, cv2.LINE_AA, False)
        image = cv2.putText(image, time_date, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 165, 255), 1, cv2.LINE_AA, False)

        # Displaying the image
        if display:
            print(f'Image displayed with timestamp: {time_date}')
            cv2.imshow(window_name, image)
            cv2.waitKey(0)

        # Saving the image
        cv2.imwrite(save_path, image)
        print(f'Image saved at {save_path}')

def parse_args_stmp():
    parser = argparse.ArgumentParser(description='run blur detection on a single image')
    parser.add_argument('-i', '--images', type=str, nargs='+', required=True, help='directory of images')
    parser.add_argument('-s', '--save-path', type=str, default=None, help='path to save output')

    parser.add_argument('-t', '--threshold', type=float, default=100.0, help='blurry threshold')
    parser.add_argument('-f', '--variable-size', action='store_true', help='fix the image size')

    parser.add_argument('-v', '--verbose', action='store_true', help='set logging level to debug')
    parser.add_argument('-d', '--display', action='store_true', help='display images')

    return parser.parse_args()

def find_images(image_paths, img_extensions=['.jpg', '.png', '.jpeg']):
    img_extensions += [i.upper() for i in img_extensions]

    for path in image_paths:
        path = pathlib.Path(path)

        if path.is_file(): # In case the argument given for image is a file
            if path.suffix not in img_extensions:
                logging.info(f'{path.suffix} is not an image extension! skipping {path}')
                continue
            else:
                #yield path
                return path

        if path.is_dir(): # In case the argument given as image is a directory
            for img_ext in img_extensions:
                #yield from path.rglob(f'*{img_ext}')
                return path.rglob(f'*{img_ext}')

if __name__ == '__main__':
    assert sys.version_info >= (3, 6), sys.version_info
    args = parse_args_stmp()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level)

    fix_size = not args.variable_size
    logging.info(f'fix_size: {fix_size}')

    Image_number = 0
    for image_path in find_images(args.images):
        Image_number = Image_number +1
        image = cv2.imread(str(image_path))
        image = stamp_image(image, save=True, save_path=f'../Images_CV/test/Image_{Image_number}.jpg')
        cv2.imshow('input', image)

        if cv2.waitKey(0) == ord('q'):
            logging.info('exiting...')
            exit()

