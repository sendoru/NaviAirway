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

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)

def main():
    sys.setrecursionlimit(123456)
    parser = argparse.ArgumentParser(description='Inference tool', fromfile_prefix_chars='@',
                                     conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument('--seg_path', nargs='+', default=[],
                        help='Segmentation file(s) to use for prediction (type:*.nii.gz)')
    parser.add_argument('--select_dir', action='store_true',
                        help='if set, consider each element in ```seg_path``` as directory and select all files in each ```seg_path``` directory')
    parser.add_argument('--save_path', type=str, required=True, default='',
                        help='File save directory')
    parser.add_argument('--image_info_csv_path', type=str, default='',
                        help='select *.csv file with image info such as image size')
    parser.add_argument('--branch_penalty', type=float, default=16.)
    parser.add_argument('--prune_threshold', type=float, default=0.05)
    parser.add_argument('--use_bfs', action='store_true')
    
    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()
        
    seg_path = []
    if not args.select_dir:
        for ph in args.seg_path:
            if ph != ''and ph[0] != '#':
                seg_path.append(ph.lstrip("./"))
    else:
        for dir in args.seg_path:
            if dir != ''and dir[0] != '#':
                file_lists = sorted(os.listdir(dir))
                for ph in file_lists:
                    seg_path.append(os.path.join(dir, ph).lstrip("./"))

    save_path = args.save_path
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except:
            pass

    image_info = pd.DataFrame()
    if args.image_info_csv_path != '':
        image_info = pd.read_csv(args.image_info_csv_path)
        image_info = image_info.set_index('path')

    generation_info = pd.DataFrame()
    generation_info_csv_path = save_path.rstrip('/').rstrip('\\') + '/' + "generation_info.csv"
    if os.path.exists(generation_info_csv_path):
        generation_info = pd.read_csv(generation_info_csv_path)

    trace_volume_by_gen_info = pd.DataFrame()
    trace_volume_by_gen_info_csv_path = save_path.rstrip('/').rstrip('\\') + '/' + "trace_volume_by_gen_info.csv"
    if os.path.exists(trace_volume_by_gen_info_csv_path):
        trace_volume_by_gen_info = pd.read_csv(trace_volume_by_gen_info_csv_path)

    trace_slice_area_info = pd.DataFrame()
    csv_path = save_path.rstrip('/').rstrip('\\') + '/' + "trace_slice_area_info.csv"
    if os.path.exists(csv_path):
        trace_slice_area_info = pd.read_csv(csv_path)

    for cur_seg_path in seg_path:
        try:
            pixdim_info = image_info.loc[cur_seg_path]
        except:
            pixdim_info = None
        break_and_save(cur_seg_path, save_path, generation_info, trace_volume_by_gen_info, trace_slice_area_info, args, pixdim_info)
        generation_info = pd.read_csv(generation_info_csv_path)
        trace_volume_by_gen_info = pd.read_csv(trace_volume_by_gen_info_csv_path)

if __name__ == "__main__":
    main()