import nibabel
import numpy as np
import pydicom
import os
from tqdm import tqdm


def convertNsave(arr, file_dir, index=0):
    dicom_file = pydicom.dcmread('images/dcmimage.dcm')
    arr = arr.astype('uint16')
    dicom_file.Rows = arr.shape[0]
    dicom_file.Columns = arr.shape[1]
    dicom_file.PhotometricInterpretation = "MONOCHROME2"
    dicom_file.SamplesPerPixel = 1
    dicom_file.BitsStored = 16
    dicom_file.BitsAllocated = 16
    dicom_file.HighBit = 15
    dicom_file.PixelRepresentation = 1
    dicom_file.PixelData = arr.tobytes()
    dicom_file.save_as(os.path.join(file_dir, f'slice{index}.dcm'))


def nifti2dicom_1file(nifti_dir, out_dir):
    nifti_file = nibabel.load(nifti_dir)
    nifti_array = nifti_file.get_fdata()
    number_slices = nifti_array.shape[2]

    for slice_ in tqdm(range(number_slices)):
        convertNsave(nifti_array[:, :, slice_], out_dir, slice_)


def nifti2dicom_mfiles(nifti_dir, out_dir=''):
    files = os.listdir(nifti_dir)
    for file in files:
        in_path = os.path.join(nifti_dir, file)
        out_path = os.path.join(out_dir, file)
        os.mkdir(out_path)
        nifti2dicom_1file(in_path, out_path)

  
input_image = "Data/STS_01/STS_01_ct_gtvt.nii.gz"
output_path = "dicom_output"
os.makedirs(output_path, exist_ok=True)
nifti2dicom_1file(input_image, output_path)