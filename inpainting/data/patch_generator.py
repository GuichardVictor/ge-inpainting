from skimage.util import view_as_windows
import numpy as np

def generate_patches(proj, mask, window_shape=(16,64,64), sliding=(10,64,64), percentage=5, nb_patches=50):
    #compute threshold from percentage
    threshold = int(window_shape[0]*window_shape[1]*window_shape[2]*percentage/100)
 
    #function used to count windows over threshold
    def is_above_threshold(arr):
        return np.sum(arr) > threshold
    
    #generate patch from window index
    def generate_patch(indices,array):
        index_min = indices*sliding
        index_max = index_min+window_shape
        return array[index_min[0]:index_max[0],index_min[1]:index_max[1],index_min[2]:index_max[2]]
        
    
    #generate windows
    mask_patches = view_as_windows(mask,window_shape,step=sliding)
    #flatten windows to count pixels
    mask_patches = mask_patches.reshape(*mask_patches.shape[:3],-1)
    #turn windows into boolean wether they contain enough mask or not
    mask_patches = np.apply_along_axis(is_above_threshold,3,mask_patches)
    #get indices of positives windows
    indices = np.argwhere(mask_patches==True)
    
    #chose random indices
    random_indices = np.random.randint(indices.shape[0], size=nb_patches)
    final_indices = indices[random_indices]
    
    #generate final patches
    patch_masks = np.apply_along_axis(generate_patch,1,final_indices,mask)
    patch_projs = np.apply_along_axis(generate_patch,1,final_indices,proj)

    return patch_masks, patch_projs