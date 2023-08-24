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

# define some constatns
CUTOFF_SLICE_COUNT = 10

def break_and_save(seg_path: str, save_path: str, generation_info: pd.DataFrame, args, pixdim_info=None):
    # read segmentation file
    print(f"Processing {seg_path}")
    seg_processed_II = sitk.GetArrayFromImage(sitk.ReadImage(seg_path)).astype(int)

    # fix unpside down
    upside_down = is_upside_down(seg_processed_II)
    if upside_down:
        seg_processed_II = seg_processed_II[-1::-1]
    seg_processed_II_clean = np.zeros_like(seg_processed_II)
    
    # fix dirty airway head
    last_nonzero = np.argwhere(np.sum(seg_processed_II, axis=(1, 2)) > 0)[-1][0]
    seg_processed_II_clean[:last_nonzero - CUTOFF_SLICE_COUNT] = seg_processed_II[:last_nonzero - CUTOFF_SLICE_COUNT]
    # The last index indicates top of slice actually
    for i in range(1, 6):
        seg_processed_II_clean[last_nonzero - CUTOFF_SLICE_COUNT - i] = \
            get_only_largest_component(seg_processed_II_clean[last_nonzero - CUTOFF_SLICE_COUNT - i])
        
    # detect tree and prune
    seg_slice_label_II, connection_dict_of_seg_II, number_of_branch_II, tree_length_II = tree_detection(seg_processed_II_clean, search_range=32, branch_penalty=args.branch_penalty, pixdim_info=pixdim_info)
    connection_dict_of_seg_II = prune_conneciton_dict(connection_dict_of_seg_II, th_ratio=args.prune_threshold)

    # assign voxel label and split left and right lung airway
    if args.use_bfs:
        voxel_by_generation = get_voxel_by_generation(seg_processed_II, connection_dict_of_seg_II)
        voxel_by_segment_no = get_voxel_by_segment_no(seg_processed_II, connection_dict_of_seg_II)
    else:
        voxel_by_generation = get_voxel_by_generation_without_bfs(seg_processed_II, connection_dict_of_seg_II)
        voxel_by_segment_no = get_voxel_by_segment_no_without_bfs(seg_processed_II, connection_dict_of_seg_II)

    voxel_by_generation_left, voxel_by_generation_right = get_left_and_right_lung_airway(voxel_by_generation, voxel_by_segment_no, connection_dict_of_seg_II)

    # revert upside down
    if upside_down:
        voxel_by_generation = voxel_by_generation[-1::-1]
        voxel_by_generation_left = voxel_by_generation_left[-1::-1]
        voxel_by_generation_right = voxel_by_generation_right[-1::-1]
        seg_processed_II = seg_processed_II[-1::-1]
        seg_processed_II_clean = seg_processed_II_clean[-1::-1]

    # resize
    scale_to = None
    if pixdim_info is not None:
        scale_to = [pixdim_info['slice_count'], 512, 512]
    if scale_to is not None:
        voxel_by_generation = transform.resize(voxel_by_generation, scale_to, order=0, mode="edge", preserve_range=True, anti_aliasing=False)
        voxel_by_generation_left = transform.resize(voxel_by_generation_left, scale_to, order=0, mode="edge", preserve_range=True, anti_aliasing=False)
        voxel_by_generation_right = transform.resize(voxel_by_generation_right, scale_to, order=0, mode="edge", preserve_range=True, anti_aliasing=False)
        seg_processed_II = transform.resize(seg_processed_II, scale_to, mode="edge", preserve_range=True, anti_aliasing=False)
        seg_processed_II_clean = transform.resize(seg_processed_II_clean, scale_to, mode="edge", preserve_range=True, anti_aliasing=False)

    # make genetion info
    if pixdim_info is None:
        voxel_size = 1
    else:
        voxel_size = pixdim_info['pixdim_x'] * pixdim_info['pixdim_y'] * pixdim_info['pixdim_z']
    dict_row = {'path' : seg_path}

    for suffix, voxel in zip(('', '_l', '_r'), (voxel_by_generation, voxel_by_generation_left, voxel_by_generation_right)):
        voxel_count_by_generation = get_voxel_count_by_generation(voxel, connection_dict_of_seg_II).astype(int)
        for j, voxel_count in enumerate(voxel_count_by_generation):
            if j == 0:
                continue
            if j == 10:
                dict_row[str(j) + suffix] = voxel_count_by_generation[j:].sum() * voxel_size
                break
            else:
                dict_row[str(j) + suffix] = voxel_count * voxel_size
    dict_row['upside_down'] = upside_down
    dict_row['has_pixdim_info'] = pixdim_info is not None
    generation_info = generation_info.append(dict_row, ignore_index=True)
    
    for i in range(0, 10):
        os.makedirs(save_path.rstrip('/').rstrip('\\') + '/high_gens/', exist_ok=True)
        seg_high_gen = (voxel_by_generation >= i).astype(np.uint8)
        sitk.WriteImage(sitk.GetImageFromArray(seg_high_gen),
                        save_path.rstrip('/').rstrip('\\')
                        + '/high_gens/'
                        + seg_path[seg_path.rfind('/') + 1:seg_path.find('.')][seg_path.rfind('\\') + 1:]
                        + f"_gen_{i + 1}_or_higher.nii.gz")
    
    # save generation info csv
    generation_info.to_csv(save_path.rstrip('/').rstrip('\\') + '/' + "generation_info.csv", index=False)
    print(generation_info)
    
    # save segmentation with generatoin labeling
    voxel_by_generation[voxel_by_generation < 0] = 0
    voxel_by_generation[voxel_by_generation > 10] = 10
    os.makedirs(save_path.rstrip('/').rstrip('\\') + '/by_gen/', exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(voxel_by_generation.astype(np.uint8)),
                    save_path.rstrip('/').rstrip('\\')
                    + '/by_gen/'
                    + seg_path[seg_path.rfind('/') + 1:seg_path.find('.')][seg_path.rfind('\\') + 1:]
                    + "_by_gen.nii.gz")
    
    voxel_by_generation_left[voxel_by_generation_left < 0] = 0
    voxel_by_generation_left[voxel_by_generation_left > 10] = 10
    os.makedirs(save_path.rstrip('/').rstrip('\\') + '/by_gen/', exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(voxel_by_generation_left.astype(np.uint8)),
                    save_path.rstrip('/').rstrip('\\')
                    + '/by_gen/'
                    + seg_path[seg_path.rfind('/') + 1:seg_path.find('.')][seg_path.rfind('\\') + 1:]
                    + "_by_gen_left.nii.gz")
    
    voxel_by_generation_right[voxel_by_generation_left < 0] = 0
    voxel_by_generation_right[voxel_by_generation_left > 10] = 10
    os.makedirs(save_path.rstrip('/').rstrip('\\') + '/by_gen/', exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(voxel_by_generation_right.astype(np.uint8)),
                    save_path.rstrip('/').rstrip('\\')
                    + '/by_gen/'
                    + seg_path[seg_path.rfind('/') + 1:seg_path.find('.')][seg_path.rfind('\\') + 1:]
                    + "_by_gen_right.nii.gz")
    
    # save the centerline result
    df_of_line_of_centerline = get_df_of_line_of_centerline(connection_dict_of_seg_II)
    fig = go.Figure()
    for item in df_of_line_of_centerline.keys():
        fig.add_trace(go.Scatter3d(x=df_of_line_of_centerline[item]["x"],
                                y=df_of_line_of_centerline[item]["y"],
                                z=df_of_line_of_centerline[item]["z"],mode='lines'))

    os.makedirs(save_path.rstrip('/').rstrip('\\') + '/centerline/', exist_ok=True)
    fig.write_html(save_path.rstrip('/').rstrip('\\')
                    + '/centerline/'
                    + seg_path[seg_path.rfind('/') + 1:seg_path.find('.')][seg_path.rfind('\\') + 1:]
                    + "_seg_result_centerline.html")

    return dict_row