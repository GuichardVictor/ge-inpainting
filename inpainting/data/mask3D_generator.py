"""
This code is not cleaned.
It is very experimental.

Generates lines and coils in a 3D volume.

Try to optimize the position of metals inside the body.

Only the Generate3DArtifacts has to be used.
"""

import math
import time
from random import randint
import numpy as np

import scipy
from skimage.measure import label
from skimage.filters import threshold_otsu

def normalize(img):
    # USED FOR NORMALIZATION
    img -= img.min()
    img /= img.max()
    return img

def normalize_vector(v):
    norm = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    
    return np.array([v[0]/norm, v[1]/norm, v[2]/norm])

def point_to_int(origin):
    return [int(round(origin[0])),  int(round(origin[1])), int(round(origin[2]))]

def check_inside(origin, shape, mask):
    r_origin = point_to_int(origin)
    if r_origin[0] < 0 or r_origin[0] >= shape[0]:
        return False
    if r_origin[1] < 0 or r_origin[1] >= shape[1]:
        return False
    if r_origin[2] < 0 or r_origin[2] >= shape[2]:
        return False
    return mask[r_origin[0], r_origin[1], r_origin[2]]


def check_inside_sphere(img, mask, origin, size):
    x = int(round(origin[0])) 
    y = int(round(origin[1])) 
    z = int(round(origin[2])) 
    if x + size < img.shape[0] and x - size >= 0 and \
       y + size < img.shape[1] and y - size >= 0 and \
       z + size < img.shape[2] and z - size >= 0:
        if mask[x + size, y, z] and  mask[x - size, y, z] and\
           mask[x, y + size, z] and  mask[x, y - size, z] and\
           mask[x, y, z + size] and  mask[x, y, z - size]:
            return True
        return False
    return False

def fill_mask(origin, mask, size, img_mask):
    start_time = time.time()
    if check_inside(np.asarray(origin + [size,size,size]), mask.shape, img_mask):
        mask[origin[0]:origin[0]+size, origin[1]:origin[1]+size, origin[2]:origin[2]+size] = True


def generate_3d_artifact(img, masks, nb_artefacts=3, len_artefacts=500, size_artefacts=10):
    coil_nb = np.random.randint(nb_artefacts, size=1)[0]
    line_nb = nb_artefacts - coil_nb
    mask = np.zeros(img.shape, dtype=bool)
    radius = size_artefacts
    
    for i in range(line_nb):
        origin = np.array([randint(0,img.shape[0]),randint(0,img.shape[1]),randint(0,img.shape[2])], dtype=np.float32)
        direction = np.array([randint(-10,10),randint(-10,10),randint(-10,10)], dtype=np.float32)
    
        direction = normalize_vector(direction)
        
        l = 0
        while (check_inside(origin, img.shape, masks) or  l < len_artefacts):
            r_origin = [int(round(origin[0])),  int(round(origin[1])), int(round(origin[2]))]
        
            fill_mask(r_origin, mask, size_artefacts, masks)
        
            origin += direction
            l += 1
            
    for i in range(coil_nb):
        origin = np.array([randint(0,img.shape[0]),randint(0,img.shape[1]),randint(0,img.shape[2])], dtype=np.float32)

        while not (check_inside_sphere(img, masks, origin, radius)):
            origin = np.array([randint(0,img.shape[0]),randint(0,img.shape[1]),randint(0,img.shape[2])], dtype=np.float32)

        print(origin)

        for i in range(radius):
            for j in range(radius - i):
                for k in range(radius - j - i):
                    mask[int(round(origin[0]))  + i][int(round(origin[1])) + j][int(round(origin[2])) + k] = True
                    mask[int(round(origin[0]))  - i][int(round(origin[1])) - j][int(round(origin[2])) - k] = True
                    mask[int(round(origin[0]))  + i][int(round(origin[1])) + j][int(round(origin[2])) - k] = True
                    mask[int(round(origin[0]))  - i][int(round(origin[1])) - j][int(round(origin[2])) + k] = True
                    mask[int(round(origin[0]))  + i][int(round(origin[1])) - j][int(round(origin[2])) + k] = True
                    mask[int(round(origin[0]))  - i][int(round(origin[1])) + j][int(round(origin[2])) - k] = True
                    mask[int(round(origin[0]))  + i][int(round(origin[1])) - j][int(round(origin[2])) - k] = True
                    mask[int(round(origin[0]))  - i][int(round(origin[1])) + j][int(round(origin[2])) + k] = True

    return mask

def create_3d_image_artifact(img, mask):
    artifacted = img.copy()
    mask = generate_3d_artifact(img, mask)
    artifacted[mask] = 1.0
    return artifacted

def fill_image_with_mask(img, mask):
    artifacted = img.copy()
    artifacted[mask] = np.max(artifacted) * 10.0
    return artifacted

def compute_mask(images):
    masks = np.zeros(shape=images.shape, dtype=np.bool)
    for i, image in enumerate(images):
        thresh = threshold_otsu(image)
        bin_img = image > thresh
        filled_image = scipy.ndimage.morphology.binary_fill_holes(bin_img)
        labels = label(filled_image)
        assert( labels.max() != 0 ) # assume at least 1 CC
        final_mask = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        masks[i] = final_mask
    return masks


def Generate3DArtifacts(im_arr):
    masks = compute_mask(im_arr)
    masks.shape, masks
    
    mask_art = generate_3d_artifact(im_arr, masks,size_artefacts=10)

    return mask_art # Project this array to get the complete data generation