import os
from random import randint, seed
import itertools
import numpy as np
import cv2

class MaskGenerator():

    def __init__(self, height : int, width : int, channels:int=1,
                scale:float=0.05, max_count:int=5, lines:bool=True,
                circles:bool=False, ellipses:bool=False, rand_seed:int=None):
        """Mask Generator class to generate masks
        
        Arguments:
            height: Mask height
            width: Mask width
        
        Keyword Arguments:
            channels: Number of channels (default: 1)
            rand_seed: Random seed (default: None)
        """

        self.height = height
        self.width = width
        self.channels = channels

        self.mask_files = []     

        self.scale = scale
        self.max_count = max_count
        self.lines = lines
        self.circles = circles
        self.ellipses = ellipses

        # Seed for reproducibility
        if rand_seed:
            seed(rand_seed)

    def _generate_mask(self, scale : float, max_count : int, lines : bool, circles : bool, ellipses : bool):
        """Generates random masks with lines, circles and ellipses"""

        img = np.zeros((self.height, self.width, self.channels), np.uint8)

        # Set size scale
        size = int((self.width + self.height) * scale)
        if self.width < 64 or self.height < 64:
            raise Exception("Mask too small (min (64, 64))")
        
        # Draw random lines
        if lines:
            for _ in range(randint(1, max_count)):
                x1, x2 = randint(1, self.width), randint(1, self.width)
                y1, y2 = randint(1, self.height), randint(1, self.height)
                thickness = randint(1, size)
                cv2.line(img,(x1,y1),(x2,y2),(1,1,1),thickness)
            
        # Draw random circles
        if circles:
            for _ in range(randint(1, max_count)):
                x1, y1 = randint(1, self.width), randint(1, self.height)
                radius = randint(3, size)
                cv2.circle(img,(x1,y1),radius,(1,1,1), -1)
            
        # Draw random ellipses
        if ellipses:
            for _ in range(randint(1, max_count)):
                x1, y1 = randint(1, self.width), randint(1, self.height)
                s1, s2 = randint(1, self.width), randint(1, self.height)    
                a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
                thickness = randint(3, size)
                cv2.ellipse(img, (x1,y1), (s1,s2), a1, a2, a3,(1,1,1), thickness)
        
        return 1-img


    def sample(self, random_seed:any=None):
        """Return a mask"""
        if random_seed:
            seed(random_seed)

        return self._generate_mask(self.scale, self.max_count, self.lines, self.circles, self.ellipses)