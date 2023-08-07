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
import skimage.transform as transform
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from .model_arch import SegAirwayModel
from .model_run import get_image_and_label, get_crop_of_image_and_label_within_the_range_of_airway_foreground, \
semantic_segment_crop_and_cat, dice_accuracy
from .post_process import post_process, add_broken_parts_to_the_result, find_end_point_of_the_airway_centerline, \
get_super_vox, Cluster_super_vox, delete_fragments, get_outlayer_of_a_3d_shape, get_crop_by_pixel_val, fill_inner_hole
from .detect_tree import tree_detection, prune_conneciton_dict
from .ulti import save_obj, load_obj, get_and_save_3d_img_for_one_case,load_one_CT_img, \
get_df_of_centerline, get_df_of_line_of_centerline
from .airway_area_utils import *

def break_and_save(seg_path: str, save_path: str, generation_info: pd.DataFrame, scale_to=None):
    CUTOFF_SLICE_COUNT = 10
    print(f"Processing {seg_path}")
    seg_processed_II = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))

    upside_down = is_upside_down(seg_processed_II)
    if upside_down:
        seg_processed_II = seg_processed_II[-1::-1]
    seg_processed_II_clean = np.zeros_like(seg_processed_II)
    
    last_nonzero = np.argwhere(np.sum(seg_processed_II, axis=(1, 2)) > 0)[-1][0]
    seg_processed_II_clean[:last_nonzero - CUTOFF_SLICE_COUNT] = seg_processed_II[:last_nonzero - CUTOFF_SLICE_COUNT]

    seg_slice_label_II, connection_dict_of_seg_II, number_of_branch_II, tree_length_II = tree_detection(seg_processed_II_clean, search_range=32)
    connection_dict_of_seg_II = prune_conneciton_dict(connection_dict_of_seg_II)

    voxel_by_generation = get_voxel_by_generation(seg_processed_II_clean, connection_dict_of_seg_II)

    voxel_by_generation[last_nonzero - CUTOFF_SLICE_COUNT:] = 2 * (seg_processed_II[last_nonzero - CUTOFF_SLICE_COUNT:] - 1)

    if upside_down:
        voxel_by_generation = voxel_by_generation[-1::-1]
        seg_processed_II = seg_processed_II[-1::-1]
        seg_processed_II_clean = seg_processed_II_clean[-1::-1]
    # voxel_count_by_generation /= voxel_count_by_generation.sum()

    if scale_to is not None:
        voxel_by_generation = np.round(transform.resize(voxel_by_generation.astype(float), scale_to, order=0)).astype(int)
        seg_processed_II = np.round(transform.resize(seg_processed_II.astype(float), scale_to)).astype(np.uint8)
        seg_processed_II_clean = np.round(transform.resize(seg_processed_II_clean.astype(float), scale_to)).astype(np.uint8)

    voxel_count_by_generation = get_voxel_count_by_generation(seg_processed_II_clean, connection_dict_of_seg_II).astype(float)
    
    df_of_line_of_centerline = get_df_of_line_of_centerline(connection_dict_of_seg_II)

    fig = go.Figure()

    for item in df_of_line_of_centerline.keys():
        fig.add_trace(go.Scatter3d(x=df_of_line_of_centerline[item]["x"],
                                y=df_of_line_of_centerline[item]["y"],
                                z=df_of_line_of_centerline[item]["z"],mode='lines'))

    # save the centerline result
    fig.write_html(save_path.rstrip('/').rstrip('\\')
                    + '/'
                    + seg_path[seg_path.rfind('/') + 1:seg_path.find('.')][seg_path.rfind('\\') + 1:]
                    + "_seg_result_centerline.html")

    dict_row = {'path' : seg_path}
    for j, ratio in enumerate(voxel_count_by_generation):
        dict_row[j + 1] = ratio
    dict_row['upside_down'] = upside_down
    generation_info = generation_info.append(dict_row, ignore_index=True)
    
    for i in range(0, 10):
        try:
            os.makedirs(save_path.rstrip('/').rstrip('\\') + '/high_gens/')
        except:
            pass
        seg_high_gen = (voxel_by_generation >= i).astype(np.uint8)
        sitk.WriteImage(sitk.GetImageFromArray(seg_high_gen),
                        save_path.rstrip('/').rstrip('\\')
                        + '/high_gens/'
                        + seg_path[seg_path.rfind('/') + 1:seg_path.find('.')][seg_path.rfind('\\') + 1:]
                        + f"_gen_{i + 1}_or_higher.nii.gz")
        
    voxel_by_generation[voxel_by_generation < 0] = -1
    voxel_by_generation += 1
    voxel_by_generation[voxel_by_generation > 6] = 6
    sitk.WriteImage(sitk.GetImageFromArray(voxel_by_generation.astype(np.uint8)),
                    save_path.rstrip('/').rstrip('\\')
                    + '/'
                    + seg_path[seg_path.rfind('/') + 1:seg_path.find('.')][seg_path.rfind('\\') + 1:]
                    + "_by_gen.nii.gz")

    generation_info.to_csv(save_path.rstrip('/').rstrip('\\') + '/' + "generation_ratio.csv", index=False)
    print(generation_info)
    return generation_info