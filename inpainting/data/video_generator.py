import os
from random import randint, seed, getrandbits
from typing import Tuple
import itertools
import numpy as np
import copy
import gc
import cv2

class ImageToVideoGenerator():

    def __init__(self, height : int, width : int, channels:int=1, count:int=5,
                 rand_seed:int=None, rotation:bool=False, shift:bool=True, transform_point:Tuple[int,int]=None,
                 tick:int=5, x_shift:int=5, y_shift:int=5):
        """Image to Video Generator class to generate videos by shifting or rotating an image

        Usage:
            video_generator = ImageToVideoGenerator(...)
            video = video_generator.sample(image)
        
        Arguments:
            height: Image height
            width: Image width
        
        Keyword Arguments:
            channels: Number of channels (default: 1)
            count: Number of additional images in the video (default: 5) (excluding reference image)
            rand_seed: Random seed (default: None)
            rotation: the video is made by rotating the image
            shift: the video is made by shifting the image (shift override rotation)
            transform_point: if transform_point is None, rotation and shift will be be applied on the center of the image else applied on (x,y)
            amount: amount of degree or pixel
            tick: degree of rotation per step
            x_shift: number of pixel shifted across x per step
            y_shift: number of pixel shifted across y per step
        """
        
        self.height = height
        self.width = width
        self.channels = channels
        self.count = count
        self.rotation = rotation
        self.shift = shift
        self.transform_point = transform_point
        self.center = (self.height / 2, self.width / 2)
        
        assert(rotation or shift)
        if transform_point:
            assert(len(transform_point) == 2)

            self.transform_point = (min(self.height - 1, max(0, self.transform_point[0])),
                                    min(self.width - 1, max(0, self.transform_point[1])))
        
        if self.shift:
            self.rotation = False
            
        self.tick = tick
        self.x_shift = x_shift
        self.y_shift = y_shift

        # Seed for reproducibility
        if rand_seed:
            seed(rand_seed)
            
    def _get_matrix_transform(self, pos_x, pos_y, step):
        """ Return the transformation matrix to be applied on image """
        M = None
        if self.shift:
                M = np.float32([[1, 0, (pos_x*2-1) * self.x_shift*(step+1)], [0, 1, (pos_y*2-1) * self.y_shift * (step+1)]])
        else:
            angle = (pos_x*2-1)*self.tick*(step+1)
            if self.transform_point is None:
                M = cv2.getRotationMatrix2D(self.center, angle, 1)
            else:
                M = cv2.getRotationMatrix2D(self.transform_point, angle, 1)
                                            
        return M

    def _generate_video(self, image):
        """Generates video from an image"""
        pos_x = int(getrandbits(1))
        pos_y = int(getrandbits(1))
        
        video = np.zeros((self.height, self.width, self.channels, self.count+1))
        video[..., 0] = image
        
        M = None

        for step in range(self.count):
            M = self._get_matrix_transform(pos_x, pos_y, step)

            res = cv2.warpAffine(copy.deepcopy(image), M, (self.height, self.width))

            video[..., step+1] = res

        gc.collect()
        
        return video


    def sample(self, image, random_seed:any=None):
        """Return a generated video"""
        if random_seed:
            seed(random_seed)

        return self._generate_video(image)