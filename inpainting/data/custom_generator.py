import copy
import gc

import numpy as np
from keras.preprocessing.image import ImageDataGenerator


class CustomDataGenerator(ImageDataGenerator):
    """ CustomDataGenerator

    Normal `ImageDataGenerator` with mask_generator
    to generate masked images.
    """

    def flow_from_directory(self, directory, mask_generator, *args, **kwargs):
        generator = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)        
        seed = None if 'seed' not in kwargs else kwargs['seed']
        
        while True:
            # Get augmentend image samples
            img = next(generator)

            # Get masks for each image sample            
            mask = np.stack([mask_generator.sample(seed) for _ in range(img.shape[0])], axis=0)

            # Apply masks to all image sample
            masked = copy.deepcopy(img) # Not copying pointers but the data
            masked[mask==0] = 1

            gc.collect()
            yield [masked, mask], img

    
    def flow(self, x, y, mask_generator, *args, **kwargs):
        generator = super().flow(x, y, class_mode=None, *args, **kwargs)
        seed = None if 'seed' not in kwargs else kwargs['seed']

        while True:
            # Get augmentend image samples
            img = next(generator)

            # Get masks for each image sample            
            mask = np.stack([mask_generator.sample(seed) for _ in range(img.shape[0])], axis=0)

            # Apply masks to all image sample
            masked = copy.deepcopy(img) # Not copying pointers but the data
            masked[mask==0] = 1

            gc.collect()
            yield [masked, mask], img