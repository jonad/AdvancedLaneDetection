import glob
import os
from preprocessing import *
import cv2
import numpy as np


def read_images(folder, file_pattern):
    images = []
    for filename in glob.glob(os.path.join(folder, file_pattern)):
        images.append(filename)
    
def save_images(img, folder, file_pattern):
    file_path = os.path.join(folder, file_pattern)
    print(file_path)
    cv2.imwrite(file_path, img)
    
    
    
print('hello')
## test grayscale function
def test_grayscale(images):
    for image in images:
        img = cv2.imread(image)
        img = grayscale(img)
        f = image.split('/')[2]
        print(f)
        save_images(img, 'testing', f)
        
def test_perspective(images):
    # pts1 = np.float32([[603, 456], [743, 463], [317, 672], [1078, 667]])
    # pts2 = np.float32([[603, 0], [743, 0], [317, 720], [1078, 720]])
    
    for image in images:
        img = cv2.imread(image)
        img_size = (img.shape[1], img.shape[0])
        src = np.float32(
            [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
             [((img_size[0] / 6) - 10), img_size[1]],
             [(img_size[0] * 5 / 6) + 60, img_size[1]],
             [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
        dst = np.float32(
            [[(img_size[0] / 4), 0],
             [(img_size[0] / 4), img_size[1]],
             [(img_size[0] * 3 / 4), img_size[1]],
             [(img_size[0] * 3 / 4), 0]])
        matrix = perspective_transform(src, dst)
        result = warp_image(img, matrix, img_size , flags=cv2.INTER_LINEAR)
        f = image.split('/')[2]
        print(f)
        save_images(result, 'testing', f)
        
        

        
        
if __name__ == '__main__':
    test_images = []
    for image in glob.glob('output_images/transform/*.jpg'):
        test_images.append(image)
    test_perspective(test_images)