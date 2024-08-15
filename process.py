import sys
import argparse
import logging
import pathlib
import json
import numpy as np

import cv2

from detection import estimate_blur
from detection import fix_image_size
from detection import pretty_blur_map
from utils import stamp_image


def parse_args():
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
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level)

    fix_size = not args.variable_size
    logging.info(f'fix_size: {fix_size}')

    if args.save_path is not None:
        save_path = pathlib.Path(args.save_path)
        assert save_path.suffix == '.json', save_path.suffix
    else:
        save_path = None

    results = []
    results_stp = []

    total_images = 0
    blurred_imgs = 0
    detected = 0

    for image_path in find_images(args.images):
        total_images = total_images + 1
        image = cv2.imread(str(image_path))
        if image is None:
            logging.warning(f'warning! failed to read image from {image_path}; skipping!')
            continue

        logging.info(f'processing {image_path}')

        if fix_size:
            image = fix_image_size(image)
        else:
            logging.warning('not normalizing image size for consistent scoring!')

        blur_map, score, blurry = estimate_blur(image, threshold=args.threshold)
        # add test to make the difference if stamped or not
        image_stp = stamp_image(image)
        _ , score_stp, blurry_stp = estimate_blur(image_stp, threshold=args.threshold)


        logging.info(f'image_path: {image_path} score: {score} blurry: {blurry}')
        logging.info(f'image_path: {image_path} score_stp: {score_stp} blurry_stp: {blurry_stp}')

        results.append({'input_path': str(image_path), 'score': score, 'blurry': blurry})
        results_stp.append({'input_path': str(image_path), 'score_stp': score_stp, 'blurry_stp': blurry_stp})
        if blurry:
            text = 'Blur Detected'
        else:
            text = 'Image Clear'

        if blurry_stp:
            text_stp = 'Blur Detected'
        else:
            text_stp = 'Image Clear'


        if str(image_path).find('blur') or str(image_path).find('Blur'):
            blurred_imgs = (blurred_imgs + 1)
            if blurry:
                detected = detected + 1
                assessement = 'Blur correctly detected'
                print(assessement)
            else:
                assessement = 'Blur not detected'
                print(assessement)


        if args.display:
            image = cv2.resize(image, (780,540), interpolation=cv2.INTER_LINEAR)
            # Add Information on the Image
            cv2.putText(image, "{}: {:.2f}".format(text, score), (570, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(image, assessement, (570, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 1, cv2.LINE_AA, False)

            # Add Information on the Image
            cv2.putText(image_stp, "{}: {:.2f}".format(text_stp, score_stp), (570, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(image_stp, assessement, (570, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 1, cv2.LINE_AA, False)

            display_image = np.hstack((image, image_stp))
            cv2.imshow('input', display_image)

            if cv2.waitKey(0) == ord('q'):
                logging.info('exiting...')
                exit()

    if save_path is not None:
        logging.info(f'saving json to {save_path}')

        with open(save_path, 'w') as result_file:
            data = {'images': args.images, 'threshold': args.threshold, 'fix_size': fix_size, 'results': results}
            json.dump(data, result_file, indent=4)

    # Show results
    print(f'The final accuracy of the model is {detected/blurred_imgs*100} %')