import pickle
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
from skimage import measure
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
from .detect_tree import tree_detection, prune_conneciton_dict,get_trace_voxel_count_by_gen_from_root, get_segment_dict
from .ulti import save_obj, load_obj, get_and_save_3d_img_for_one_case,load_one_CT_img, \
get_df_of_centerline, get_df_of_line_of_centerline
from .airway_area_utils import *

# define some constatns
CUTOFF_SLICE_COUNT = 10

def break_and_save(seg_path: str, save_path: str, generation_info: pd.DataFrame, trace_volume_by_gen_info: pd.DataFrame, trace_slice_area_info: pd.DataFrame, args, pixdim_info=None):

    generation_info = pd.DataFrame()
    csv_path = save_path.rstrip('/').rstrip('\\') + '/' + "generation_info.csv"
    if os.path.exists(csv_path):
        generation_info = pd.read_csv(csv_path)

    trace_volume_by_gen_info = pd.DataFrame()
    csv_path = save_path.rstrip('/').rstrip('\\') + '/' + "trace_volume_by_gen_info.csv"
    if os.path.exists(csv_path):
        trace_volume_by_gen_info = pd.read_csv(csv_path)

    trace_slice_area_info = pd.DataFrame()

    # read segmentation file
    print(f"Processing {seg_path}")
    seg_processed_II = sitk.GetArrayFromImage(sitk.ReadImage(seg_path)).astype(int)
    extended_image_height = seg_processed_II.shape[0]

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

    segment_dict = get_segment_dict(connection_dict_of_seg_II, pixdim_info)

    voxel_by_generation_left, voxel_by_generation_right = get_left_and_right_lung_airway(voxel_by_generation, voxel_by_segment_no, connection_dict_of_seg_II, segment_dict)

    # get segment dict and find voxel count by generation on each trace starting from root
    voxel_count_by_segment_no = get_voxel_count_by_segment_no(voxel_by_segment_no, segment_dict)
    segment_dict = get_trace_voxel_count_by_gen_from_root(segment_dict, voxel_count_by_segment_no)

    # revert upside down
    if upside_down:
        voxel_by_generation = voxel_by_generation[-1::-1]
        voxel_by_generation_left = voxel_by_generation_left[-1::-1]
        voxel_by_generation_right = voxel_by_generation_right[-1::-1]
        seg_processed_II = seg_processed_II[-1::-1]
        seg_processed_II_clean = seg_processed_II_clean[-1::-1]
        for key, val in segment_dict.items():
            segment_dict[key]['endpoint_loc'][0] = seg_processed_II.shape[0] - segment_dict[key]['endpoint_loc'][0] - 1
        for key, val in connection_dict_of_seg_II.items():
            connection_dict_of_seg_II[key]['loc'][0] = seg_processed_II.shape[0] - connection_dict_of_seg_II[key]['loc'][0] - 1

    # resize
    scale_to = None
    seg_processed_II_extended = seg_processed_II.copy()
    if pixdim_info is not None:
        scale_to = [pixdim_info['slice_count'], 512, 512]
    if scale_to is not None:
        voxel_by_generation = transform.resize(voxel_by_generation, scale_to, order=0, mode="edge", preserve_range=True, anti_aliasing=False)
        voxel_by_generation_left = transform.resize(voxel_by_generation_left, scale_to, order=0, mode="edge", preserve_range=True, anti_aliasing=False)
        voxel_by_generation_right = transform.resize(voxel_by_generation_right, scale_to, order=0, mode="edge", preserve_range=True, anti_aliasing=False)
        seg_processed_II = transform.resize(seg_processed_II, scale_to, mode="edge", preserve_range=True, anti_aliasing=False)
        seg_processed_II_clean = transform.resize(seg_processed_II_clean, scale_to, mode="edge", preserve_range=True, anti_aliasing=False)

    # make genetion info dataframe
    if pixdim_info is None:
        voxel_size = 1
    else:
        voxel_size = pixdim_info['pixdim_x'] * pixdim_info['pixdim_y'] * pixdim_info['pixdim_z']
    dict_row = {'path' : seg_path}

    for suffix, voxel in zip(('total', 'l', 'r'), (voxel_by_generation, voxel_by_generation_left, voxel_by_generation_right)):
        voxel_count_by_generation = get_voxel_count_by_generation(voxel, connection_dict_of_seg_II).astype(int)
        for j, voxel_count in enumerate(voxel_count_by_generation):
            if j == 0:
                continue
            if j == 10:
                # dict_row[str(j) + suffix] = voxel_count_by_generation[j:].sum() * voxel_size
                dict_row['_'.join((str(j), 'volume', suffix))] = voxel_count_by_generation[j] * voxel_size
                break
            else:
                dict_row['_'.join((str(j), 'volume', suffix))] = voxel_count * voxel_size

        dict_row['volume_sum_' + suffix] = voxel_count_by_generation[1:11].sum() * voxel_size

        for j, voxel_count in enumerate(voxel_count_by_generation):
            if j == 0:
                continue
            if j == 10:
                # dict_row[str(j) + suffix] = voxel_count_by_generation[j:].sum() * voxel_size
                dict_row['_'.join((str(j), 'volume_ratio', suffix))] = voxel_count_by_generation[j] / voxel_count_by_generation[1:11].sum()
                break
            else:
                dict_row['_'.join((str(j), 'volume_ratio', suffix))] = voxel_count / voxel_count_by_generation[1:11].sum()

        branch_count = [0 for _ in range(11)]

        if suffix == 'total':
            for key, val in segment_dict.items():
                branch_count[val['generation']] += 1
        elif suffix == 'l':
            for key, val in segment_dict.items():
                if val['side'] == 'left':
                    branch_count[val['generation']] += 1
        elif suffix == 'r':
            for key, val in segment_dict.items():
                if val['side'] == 'right':
                    branch_count[val['generation']] += 1

        for j in range(1, 11):
            dict_row['_'.join((str(j), 'branch_count', suffix))] = branch_count[j]
    dict_row['upside_down'] = upside_down
    dict_row['has_pixdim_info'] = pixdim_info is not None

    # 23-12-20 요청사항
    # --------------------------------
    last_branch_observed = 0
    for suffix, voxel in zip(('_total', '_l', '_r'), (voxel_by_generation, voxel_by_generation_left, voxel_by_generation_right)):
        voxel_count_by_generation = get_voxel_count_by_generation(voxel, connection_dict_of_seg_II).astype(int)

        if suffix == '_total':
            for j, voxel_count in enumerate(voxel_count_by_generation):
                if j == 0:
                    continue
                else:
                    if voxel_count_by_generation[j+1:11].sum() < 0.001:
                        last_branch_observed = j
                        break
            dict_row['last_branch_observed'] = last_branch_observed
        branch_count = 0

        if suffix == '_total':
            for key, val in segment_dict.items():
                if val['generation'] == last_branch_observed:
                    branch_count += 1
        elif suffix == '_l':
            for key, val in segment_dict.items():
                if val['generation'] == last_branch_observed and val['side'] == 'left':
                    branch_count += 1
        elif suffix == '_r':
            for key, val in segment_dict.items():
                if val['generation'] == last_branch_observed and val['side'] == 'right':
                    branch_count += 1

        dict_row["no_of_bronchlole_at_the_last_branch" + suffix] = branch_count
        dict_row["vol_of_bronchlole_at_the_last_branch" + suffix] = voxel_count_by_generation[last_branch_observed] * voxel_size

    # --------------------------------
        
    # 24-01-03 요청사항 (1) 
    # --------------------------------
    voxel_count_by_generation_left = get_voxel_count_by_generation(voxel_by_generation_left, connection_dict_of_seg_II).astype(int)
    dict_row["last_branch_observed_l"] = (voxel_count_by_generation_left > 0).astype(int).sum() + 1
    voxel_count_by_generation_right = get_voxel_count_by_generation(voxel_by_generation_right, connection_dict_of_seg_II).astype(int)
    dict_row["last_branch_observed_r"] = (voxel_count_by_generation_right > 0).astype(int).sum() + 1
    
    # --------------------------------

    generation_info = generation_info.append(dict_row, ignore_index=True)

    # make trace_volume_by_gen info dataframe
    # TODO 저기 10이라는 숫자 변수로 바꿔서 뭔가뭔가 하게
    seg_processed_II_extended_labeled = np.zeros_like(seg_processed_II_extended)
    for i in range(len(seg_processed_II_extended)):
        seg_processed_II_extended_labeled[i] = measure.label(seg_processed_II_extended[i], connectivity=1)

    for key, val in segment_dict.items():
        if val['generation'] == 10 or (val['generation'] < 10 and len(val['next']) == 0):
            dict_row = {'path' : seg_path}
            dict_row['highest_generation'] = val['generation']
            dict_row['x'] = val['endpoint_loc'][1]
            dict_row['y'] = val['endpoint_loc'][2]
            dict_row['z'] = val['endpoint_loc'][0]
            if pixdim_info is not None:
                one_pixel_area = pixdim_info['pixdim_x'] * pixdim_info['pixdim_y']
            else:
                one_pixel_area = 1.
            dict_row['endpoint_area'] = (seg_processed_II_extended_labeled[val['endpoint_loc'][0]] == seg_processed_II_extended_labeled[val['endpoint_loc'][0], val['endpoint_loc'][1], val['endpoint_loc'][2]]).sum() * one_pixel_area
            dict_row['endpoint_diameter'] = 2 * np.sqrt(dict_row['endpoint_area'] / np.pi)

            volume_sum = 0.
            for i, cnt in enumerate(val['trace_voxel_count_by_gen']):
                if i != 0:
                    dict_row[f'{str(i)}'] = cnt * voxel_size
                    if pixdim_info is not None:
                        dict_row[f'{str(i)}'] *= pixdim_info['slice_count'] / extended_image_height
                    volume_sum += dict_row[f'{str(i)}']
            
            for i, cnt in enumerate(val['trace_voxel_count_by_gen']):
                if i != 0:
                    dict_row[f'{str(i)}_ratio'] = dict_row[f'{str(i)}'] / volume_sum

            # trace_volume_by_gen_info = pd.DataFrame(trace_volume_by_gen_info.append(dict_row, ignore_index=True))
            trace_volume_by_gen_info = trace_volume_by_gen_info.append(dict_row, ignore_index=True)

    for key, val in segment_dict.items():
        if val['generation'] == 10 or (val['generation'] < 10 and len(val['next']) == 0):
            dict_row_base = {'path' : seg_path}
            dict_row_base['highest_generation'] = val['generation']
            dict_row_base['endpoint_x'] = val['endpoint_loc'][1]
            dict_row_base['endpoint_y'] = val['endpoint_loc'][2]
            dict_row_base['endpoint_z'] = val['endpoint_loc'][0]
            cur_point = val['endpoint']
            dist_from_endpoint = 0
            while True:
                dict_row = dict_row_base.copy()
                cur_point_loc = connection_dict_of_seg_II[cur_point]['loc']
                dict_row['generation'] = connection_dict_of_seg_II[cur_point]['generation']
                dict_row['x'] = cur_point_loc[1]
                dict_row['y'] = cur_point_loc[2]
                dict_row['z'] = cur_point_loc[0]
                dict_row['dist_from_endpoint'] = dist_from_endpoint
                dict_row['slice_area'] = \
                    (seg_processed_II_extended_labeled[cur_point_loc[0]] == seg_processed_II_extended_labeled[cur_point_loc[0], cur_point_loc[1], cur_point_loc[2]]).sum() \
                        * pixdim_info['pixdim_x'] * pixdim_info['pixdim_y']
                dict_row['slice_diameter'] = 2 * np.sqrt(dict_row['slice_area'] / np.pi)

                trace_slice_area_info = trace_slice_area_info.append(dict_row, ignore_index=True)

                if len(connection_dict_of_seg_II[cur_point]['before']) == 0 or connection_dict_of_seg_II[cur_point]['before'][0] == 0:
                    break
                cur_point = connection_dict_of_seg_II[cur_point]['before'][0]
                dist_from_endpoint += 1
    
    # save generation info csv
    generation_info.to_csv(save_path.rstrip('/').rstrip('\\') + '/' + "generation_info.csv", index=False)
    print(generation_info)

    # save trace volume csv
    trace_volume_by_gen_info.to_csv(save_path.rstrip('/').rstrip('\\') + '/' + "trace_volume_by_gen_info.csv", index=False)
    print(trace_volume_by_gen_info)

    # trace_slice_area_info
    seg_file_name = seg_path
    seg_file_name.replace('\\', '/')
    seg_file_name = seg_file_name.split('/')[-1]
    seg_file_name = seg_file_name.split('.')[0]
    os.makedirs(save_path.rstrip('/').rstrip('\\') + f"/trace_slice_area_info/", exist_ok=True)
    trace_slice_area_info.to_csv(save_path.rstrip('/').rstrip('\\') + f"/trace_slice_area_info/" + f"trace_slice_area_info_{seg_file_name}.csv", index=False)
    print(trace_slice_area_info)

    os.makedirs(save_path.rstrip('/').rstrip('\\') + '/centerline_dicts/', exist_ok=True)
    centerline_dict_path = \
        save_path.rstrip('/').rstrip('\\') + '/centerline_dicts/' + seg_path[seg_path.rfind('/') + 1:seg_path.find('.')][seg_path.rfind('\\') + 1:] + '.pkl'
    with open(centerline_dict_path, 'wb') as f:
        pickle.dump(connection_dict_of_seg_II, f)


    for i in range(0, 10):
        os.makedirs(save_path.rstrip('/').rstrip('\\') + '/high_gens/', exist_ok=True)
        seg_high_gen = (voxel_by_generation >= i).astype(np.uint8)
        sitk.WriteImage(sitk.GetImageFromArray(seg_high_gen),
                        save_path.rstrip('/').rstrip('\\')
                        + '/high_gens/'
                        + seg_path[seg_path.rfind('/') + 1:seg_path.find('.')][seg_path.rfind('\\') + 1:]
                        + f"_gen_{i + 1}_or_higher.nii.gz")

    # save segmentation with generatoin labeling
    voxel_by_generation[voxel_by_generation < 0] = 0
    # originally ... > 10] = 0
    voxel_by_generation[voxel_by_generation > 10] = 0
    os.makedirs(save_path.rstrip('/').rstrip('\\') + '/segment_by_gen/', exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(voxel_by_generation.astype(np.uint8)),
                    save_path.rstrip('/').rstrip('\\')
                    + '/segment_by_gen/'
                    + seg_path[seg_path.rfind('/') + 1:seg_path.find('.')][seg_path.rfind('\\') + 1:]
                    + "_by_gen.nii.gz")
    
    voxel_by_generation_left[voxel_by_generation_left < 0] = 0
    # originally ... > 10] = 0
    voxel_by_generation_left[voxel_by_generation_left > 10] = 0
    sitk.WriteImage(sitk.GetImageFromArray(voxel_by_generation_left.astype(np.uint8)),
                    save_path.rstrip('/').rstrip('\\')
                    + '/segment_by_gen/'
                    + seg_path[seg_path.rfind('/') + 1:seg_path.find('.')][seg_path.rfind('\\') + 1:]
                    + "_by_gen_left.nii.gz")
    
    voxel_by_generation_right[voxel_by_generation_left < 0] = 0
    # originally ... > 10] = 0
    voxel_by_generation_right[voxel_by_generation_left > 10] = 0
    sitk.WriteImage(sitk.GetImageFromArray(voxel_by_generation_right.astype(np.uint8)),
                    save_path.rstrip('/').rstrip('\\')
                    + '/segment_by_gen/'
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