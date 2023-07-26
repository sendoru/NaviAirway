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

from func.model_arch import SegAirwayModel
from func.model_run import get_image_and_label, get_crop_of_image_and_label_within_the_range_of_airway_foreground, \
semantic_segment_crop_and_cat, dice_accuracy
from func.post_process import post_process, add_broken_parts_to_the_result, find_end_point_of_the_airway_centerline, \
get_super_vox, Cluster_super_vox, delete_fragments, get_outlayer_of_a_3d_shape, get_crop_by_pixel_val, fill_inner_hole
from func.detect_tree import tree_detection
from func.ulti import save_obj, load_obj, get_and_save_3d_img_for_one_case,load_one_CT_img, \
get_df_of_centerline, get_df_of_line_of_centerline
from func.airway_area_utils import *

def main():
    parser = argparse.ArgumentParser(description='Inference tool', fromfile_prefix_chars='@',
                                     conflict_handler='resolve')
    parser.add_argument('--weight_path', nargs='+', default=[],
                        help="weight file(s) to use for prediction")
    parser.add_argument('--image_path', nargs='+', default=[],
                        help='Image file(s) to use for prediction (type:*.nii.gz)')
    parser.add_argument('--save_path', type=str, required=True, default='',
                        help='File save directory')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Threshold probability value to decide if a voxel is included in airway or not')
    weight_path = []

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    for ph in args.weight_path:
        if ph != '' and ph[0] != '#':
            weight_path.append(ph)
    
    image_path = []
    for ph in args.image_path:
        if ph != ''and ph[0] != '#':
            image_path.append(ph)

    save_path = args.save_path
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except:
            pass

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    threshold = args.threshold
    generation_ratio = pd.DataFrame()

    models = [None for _ in range(len(weight_path))]

    for i, load_path in enumerate(weight_path):
        models[i]=SegAirwayModel(in_channels=1, out_channels=2)
        models[i].to(device)
        checkpoint = torch.load(load_path)
        models[i].load_state_dict(checkpoint['model_state_dict'])

    for image_path in image_path:
        raw_img = load_one_CT_img(image_path)
        seg_result_comb = np.zeros(raw_img.shape, dtype=float)
        seg_onehot_comb = np.zeros(raw_img.shape, dtype=int)

        seg_processed_II = seg_onehot_comb
        for i, load_path in enumerate(weight_path):
            seg_result = \
            semantic_segment_crop_and_cat(raw_img, models[i], device,
            crop_cube_size=[32, 128, 128], stride=[16, 64, 64], windowMin=-1000, windowMax=600)
            seg_result_comb += seg_result
            seg_onehot_comb += np.array(seg_result>threshold, dtype=int)
        seg_result_comb /= len(weight_path)
        seg_onehot_comb = np.array(seg_onehot_comb>0, dtype=int)
        seg_processed,_ = post_process(seg_onehot_comb, threshold=threshold)
        seg_slice_label_I, connection_dict_of_seg_I, number_of_branch_I, tree_length_I = tree_detection(seg_processed, search_range=2)
        seg_processed_II = add_broken_parts_to_the_result(connection_dict_of_seg_I, seg_result_comb, seg_processed, threshold = threshold,
                                                    search_range = 10, delta_threshold = 0.05, min_threshold = 0.4)
        seg_slice_label_II, connection_dict_of_seg_II, number_of_branch_II, tree_length_II = tree_detection(seg_processed_II, search_range=2)

        voxel_by_generation = get_voxel_by_generation(seg_processed_II, connection_dict_of_seg_II)
        voxel_count_by_generation = get_voxel_count_by_generation(seg_processed_II, connection_dict_of_seg_II).astype(float)
        voxel_count_by_generation /= voxel_count_by_generation.sum()

        dict_row = {'path' : image_path}
        for j, ratio in enumerate(voxel_count_by_generation):
            dict_row[j] = ratio

        generation_ratio.append(dict_row, ignore_index=True)

        sitk.WriteImage(sitk.GetImageFromArray(seg_processed_II),
                        save_path.rstrip('/').rstrip('\\')
                        + '/'
                        + image_path[image_path.rfind('/') + 1:image_path.find('.')][image_path.rfind('\\') + 1:]
                        + "_segmentation.nii.gz")
        for i in range(1, 10):
            seg_high_gen = (voxel_by_generation >= i).astype(int)
            sitk.WriteImage(sitk.GetImageFromArray(seg_high_gen),
                            save_path.rstrip('/').rstrip('\\')
                            + '/'
                            + image_path[image_path.rfind('/') + 1:image_path.find('.')][image_path.rfind('\\') + 1:]
                            + f"_segmentation_gen_{i}_or_higher.nii.gz")

    generation_ratio.to_csv(save_path + save_path.rstrip('/').rstrip('\\') + os.sep + "generation_ratio.csv")

if __name__ == "__main__":
    main()