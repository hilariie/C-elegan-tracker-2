import cv2
import skimage.filters as filters
import os
import numpy as np

def convert_images(path):
    print(f'processing for {path[-12:-7]}')
    images = os.listdir(path)
    for image in images:
        new_image = cv2.imread(f'{path}/{image}')
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        threshold = filters.threshold_local(new_image, block_size=101, offset=8)
        new_image = new_image < threshold
        new_image = np.where(new_image, 255, 0).astype(np.uint8)
        cv2.imwrite(f'{path[:-7]}/gray_images/{image}', new_image)
#convert_images(train_path)
#convert_images(val_path)

if __name__ == '__main__':
    import time
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    print('starting')
    train_path = 'data/train_images/images/train/images'
    val_path = 'data/train_images/images/valid/images'
    t1 = time.time()
    with ThreadPoolExecutor() as executor:
        executor.submit(convert_images, train_path)
        executor.submit(convert_images, val_path)
    print(time.time() - t1)