import cv2
import skimage.filters as filters
import os
import numpy as np

def convert_images(path):
    print(f'processing for {path[9:14]}')
    images = os.listdir(path)
    #img = cv2.imread(f'{path}/{images[4]}')
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #threshold = filters.threshold_local(img, block_size=101, offset=12)
    for image in images:
        new_image = cv2.imread(f'{path}/{image}')
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        # if 'out20002' or 'out20003' in image:
        threshold = filters.threshold_local(new_image, block_size=101, offset=12)
        new_image = new_image < threshold
        new_image = np.where(new_image, 255, 0).astype(np.uint8)
        cv2.imwrite(f'{path[:14]}/gray_images/{image}', new_image)
#convert_images(train_path)
#convert_images(val_path)

if __name__ == '__main__':
    import time
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    print('starting')
    train_path = 'data/gry/train/.imagesorig'
    val_path = 'data/gry/valid/.images_orig'
    t1 = time.time()
    with ThreadPoolExecutor() as executor:
        executor.submit(convert_images, train_path)
        executor.submit(convert_images, val_path)
    print(time.time() - t1)