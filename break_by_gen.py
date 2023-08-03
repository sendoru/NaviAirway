import numpy as np
import torch
import copy
import pandas as pd
import SimpleITK as sitk
from PIL import Image
import pydicom
import cv2
import nibabel as nib
import os
import sys
import skimage.io as io
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from func.break_and_save_utils import break_and_save
from func.model_arch import SegAirwayModel
from func.model_run import get_image_and_label, get_crop_of_image_and_label_within_the_range_of_airway_foreground, \
semantic_segment_crop_and_cat, dice_accuracy
from func.post_process import post_process, add_broken_parts_to_the_result, find_end_point_of_the_airway_centerline, \
get_super_vox, Cluster_super_vox, delete_fragments, get_outlayer_of_a_3d_shape, get_crop_by_pixel_val, fill_inner_hole
from func.ulti import save_obj, load_obj, get_and_save_3d_img_for_one_case,load_one_CT_img, \
get_df_of_centerline, get_df_of_line_of_centerline
from func.airway_area_utils import *

def main():
    CUTOFF_SLICE_COUNT = 10
    sys.setrecursionlimit(123456)
    parser = argparse.ArgumentParser(description='Inference tool', fromfile_prefix_chars='@',
                                     conflict_handler='resolve')
    parser.add_argument('--seg_path', nargs='+', default=[],
                        help='Segmentation file(s) to use for prediction (type:*.nii.gz)')
    parser.add_argument('--save_path', type=str, required=True, default='',
                        help='File save directory')
    
    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()
        
    seg_path = []
    for ph in args.seg_path:
        if ph != ''and ph[0] != '#':
            seg_path.append(ph)

    save_path = args.save_path
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except:
            pass

    generation_info = pd.DataFrame()
    csv_path = save_path.rstrip('/').rstrip('\\') + '/' + "generation_info.csv"
    if os.path.exists(csv_path):
        generation_info = pd.read_csv(csv_path)

    for cur_seg_path in seg_path:
        generation_info = break_and_save(cur_seg_path, save_path, generation_info)

if __name__ == "__main__":
    main()