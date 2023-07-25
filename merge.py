import os
import numpy as np
import SimpleITK as sitk
from PIL import Image
import pydicom
import cv2
import nibabel as nib

def loadFile(filename):
    ds = sitk.ReadImage(filename)
    img_array = sitk.GetArrayFromImage(ds)
    frame_num, width, height = img_array.shape
    return img_array, frame_num, width, height

def get_3d_img_for_one_case(img_path_list):
    img_3d=[]
    for idx, img_path in enumerate(img_path_list):
        print("progress: "+str(idx/len(img_path_list))+"; "+str(img_path), end="\r")
        img_slice, frame_num, _, _ = loadFile(img_path)
        assert frame_num==1
        img_3d.append(img_slice)
    img_3d=np.array(img_3d)
    return img_3d.reshape(img_3d.shape[0], img_3d.shape[2], img_3d.shape[3])

orig_file_path = "./data/Airway/EXACT09_3D/train_label_r/CASE01_label.nii.gz"
new_file_path = "./data/Airway/EXACT09_3D/train_label_r/CASE01_label_r0.nii.gz"
result_file_path = "./data/Airway/EXACT09_3D/train_label_r/CASE01_label_r.nii.gz"

orig_img = loadFile(orig_file_path)[0]
new_img = loadFile(new_file_path)[0]

result_img = ((orig_img + new_img) >= 1).astype(orig_img.dtype)
sitk.WriteImage(sitk.GetImageFromArray(result_img), result_file_path)