import argparse
import sys 
import os

import numpy as np 
import nibabel as nib

try:
    from keras import backend as K
    from keras.models import *
except:
    print('ERROR: Failed to initialize tensorflow-gpu and Keras. Please ensure that this module is installed and that a GPU is ready accessible.')
    sys.exit(1)

from defacer_utils import resize_img, dice_coefficient, resample_image, pre_process_image, get_available_gpus

def deface_3D_MRI():

    if len(sys.argv) < 1:
        print "usage: Please specify the filepath of a MRI image for defacing....(e.g deepdefacer <path of MRI>"
        sys.exit(1) 

    if not get_available_gpus:
        print("ERROR: Could not find an available GPU on your system. Please check that your GPU drivers (cudaNN, etc) are up to date and accessible.")
        sys.exit(1) 

    MRI_image_path = sys.argv[1]
    if '.nii' not in MRI_image_path or '.nii.gz' not in MRI_image_path:
        print("ERROR: Please ensure that input MRI file is in .nii or .nii.gz format")

    print('Preproessing input MRI image...')

    MRI_image_shape = nib.load(MRI_image_path).get_data().shape

    MRI_image_data, MRI_unnormalized_data = pre_process_image(MRI_image_path)

    deepdeface_model = load_model('deepdefacer/model.hdf5', custom_objects={'dice_coefficient': dice_coefficient})

    print('Masking %s ....' % (MRI_image_path))

    mask_prediction = deepdeface_model.predict(MRI_image_data) 

    mask_prediction[mask_prediction < 0.5] = 0 
    mask_prediction[mask_prediction >= 0.5] = 1
 
    mask_prediction = np.squeeze(mask_prediction) 

    masked_image = np.multiply(MRI_unnormalized_data, mask_prediction)

    masked_image_save = nib.Nifti1Image(masked_image, nib.load(MRI_image_path).affine)
 
    masked_image_resampled = resample_image(masked_image_save, target_shape=MRI_image_shape, get_nifti=True)

    output_file = os.path.splitext(os.path.splitext(os.path.basename(MRI_image_path))[0])[0] + '_defaced.nii.gz'


    print('Completed! Saving to %s...' % (output_file))

    nib.save(masked_image_resampled, output_file)

    print('Saved.') 












