from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import mean_absolute_error as mae
from skimage.metrics import peak_signal_noise_ratio as psnr

import numpy as np

#apply metrics on each image, average to return an array, otherwise the mean
def compute_metric(y, y_pred, mask, idx, input_shape, average=True):
    ssim_values = np.zeros(y.shape[0])
    mse_values = np.zeros(y.shape[0])
    mae_values = np.zeros(y.shape[0])
    psnr_values = np.zeros(y.shape[0])
    
    shape = y[0].shape

    for i in range(y.shape[0]):
        y_ = y[i]
        pred = y_pred[i]
        comp =  np.where(mask[i] == 1, pred[idx, ...], y_[idx]).reshape((shape[0], shape[1]))

        ssim_values[i] = ssim(y_, comp)
        mse_values[i] = mse(y_, comp)
        mae_values[i] = mae(y_, comp)
        psnr_values[i] = psnr(y_, comp)

    if average:
        return np.mean(ssim), np.mean(mse), np.mean(mae), np.mean(psnr)
    
    return ssim, mse, mae, psnr