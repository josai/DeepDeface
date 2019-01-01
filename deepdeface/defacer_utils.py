import numpy as np
import nibabel as nib

from keras import backend as K
from keras.models import *
from tensorflow.python.client import device_lib
from nilearn.image import resample_img

def resize_img(img,target_shape,mask=False, pad=False):
    ''' Resample image to specified target shape '''
    # Define interpolation method
    interp = 'nearest' if mask else 'continuous'
    if not pad:
        # Define resolution
        img_shape = np.array(img.shape[:3])
        target_shape = np.array(target_shape)
        res = img_shape/target_shape
        # Define target affine matrix
        new_affine = np.zeros((4,4))
        new_affine[:3,:3] = np.diag(res)
        new_affine[:3,3] = target_shape*res/2.*-1

        new_affine[3,3] = 1.
        
        # Resample image w. defined parameters
        reshaped_img = resample_img(img,
                                    target_affine=new_affine,
                                    target_shape=target_shape,
                                    interpolation=interp)
    else: # padded/cropped image
        reshaped_img = resample_img(img, 
                                    target_affine=img.affine,
                                    target_shape=target_shape,
                                    interpolation=interp)
    return reshaped_img

''' Dice Coefficient Metric '''
def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def resample_image(img_data, target_shape, get_nifti=False):

    img = resize_img(img_data, target_shape=target_shape, mask=False, pad=True) #resample. 

    if get_nifti:
        return img 

    return img.get_data()


def pre_process_image(img_file):
 
    nifti_image = nib.load(img_file)

    img_data = resample_image(nifti_image, (256,320,256)) 

    resamp_img = np.squeeze(img_data.astype(np.float32))

    img_data = np.expand_dims(resamp_img, axis=0) 

    img_data = np.expand_dims(img_data, axis=0) 

    min_val = np.min(img_data) 
    max_val = np.max(img_data)

    norm_img_data = (img_data - min_val) / (max_val - min_val + 1e-7) 
    return norm_img_data, resamp_img

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']