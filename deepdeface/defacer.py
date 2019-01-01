import numpy as np 
import nibabel as nib
import argparse
import sys 
import pdb
import os

from keras import backend as K
from keras.models import *

from defacer_utils import resize_img, dice_coefficient, resample_img, pre_process_img

def deface_3D_MRI(MRI_image_path):

    print('Preproessing input MRI image...')

    MRI_image_shape = nib.load(MRI_image_path).get_data().shape

    MRI_image_data, MRI_unnormalized_data = pre_process_image(MRI_image_path)

    deepdeface_model = load_model('model.hdf5', custom_objects={'dice_coefficient': dice_coefficient})

    print('Masking %s ....' % (MRI_image))

    mask_prediction = deepdeface_model.predict(MRI_image_data) 

    mask_prediction[mask_prediction < 0.5] = 0 
    mask_prediction[mask_prediction >= 0.5] = 1
 
    mask_prediction = np.squeeze(mask_prediction) 

    masked_image = np.multiply(MRI_unnormalized_data, mask_prediction)

    masked_image_save = nib.Nifti1Image(masked_image, nib.load(MRI_image).affine)
 
    masked_image_resampled = resample_image(masked_image_save, target_shape=MRI_image_shape, get_nifti=True)

    output_file = os.path.splitext(os.path.splitext(os.path.basename(MRI_image))[0])[0] + '_defaced.nii.gz'


    print('Completed! Saving to %s...' % (output_file))

    nib.save(masked_image_resampled, output_file)

    print('Saved.') 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process input images')
    parser.add_argument("input_file")


    args = parser.parse_args()

    if not args.input_file:
        print('Please specify the path of a MRI image for defacing.')
        sys.exit()


    MRI_image_path = args.input_file

    deface_3D_MRI(MRI_image_path)

   












