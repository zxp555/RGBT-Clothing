import cv2
import numpy as np
from tqdm import tqdm
import os
import shutil

def is_daytime(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    average_brightness = np.mean(gray)
    threshold = 150
    return average_brightness > threshold, average_brightness

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8,8)):
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_clahe = clahe.apply(l)
        lab = cv2.merge((l_clahe, a, b))
        enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_image = clahe.apply(image)
    return enhanced_image

def batch_process_images(src_dir, dst_dir, file_extension='.jpg'):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for filename in tqdm(os.listdir(src_dir)):
        if filename.lower().endswith(file_extension):
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(dst_dir, filename)
            img = cv2.imread(src_path)
            if img is not None:
                if not is_daytime(src_path)[0]:
                    shutil.copy(src_path, dst_path)
                    print(f'Copy finished: {filename}')
                    continue
                enhanced = apply_clahe(img)
                cv2.imwrite(dst_path, enhanced)
                print(f'Process finished: {filename}')
            else:
                print(f'Cannot read image: {filename}')

src_dir = 'rgb-train'  
dst_dir = 'rgb-enhanced-train'  
batch_process_images(src_dir, dst_dir)

src_dir = 'rgb-test'  
dst_dir = 'rgb-enhanced-test'  
batch_process_images(src_dir, dst_dir)

src_dir = 'rgb-val'  
dst_dir = 'rgb-enhanced-val'  
batch_process_images(src_dir, dst_dir)