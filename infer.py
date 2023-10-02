import time
import datetime
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
import skimage.transform as transform
import skimage.io as io
import argparse
import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from func.model_arch import SegAirwayModel
from func.model_run import get_image_and_label, get_crop_of_image_and_label_within_the_range_of_airway_foreground, \
semantic_segment_crop_and_cat, dice_accuracy
from func.post_process import post_process, post_process_v2, add_broken_parts_to_the_result, find_end_point_of_the_airway_centerline, \
get_super_vox, Cluster_super_vox, delete_fragments, get_outlayer_of_a_3d_shape, get_crop_by_pixel_val, fill_inner_hole
from func.detect_tree import tree_detection
from func.ulti import save_obj, load_obj, get_and_save_3d_img_for_one_case,load_one_CT_img, \
get_df_of_centerline, get_df_of_line_of_centerline
from func.airway_area_utils import *
from func.break_and_save_utils import break_and_save

np.int = int

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)

def main():
    # ---------- init configs ----------
    MIN_SLICE_COUNT = 200
    cur_time_str = datetime.datetime.now().__str__().replace(' ', '_').replace(':', '-')
    cur_time_str = cur_time_str[:cur_time_str.rfind('.')]
    logging.basicConfig(filename=f"results/{cur_time_str}.log", level=logging.INFO)
    sys.setrecursionlimit(100000)

    # ---------- argparsing ----------
    parser = argparse.ArgumentParser(description='Inference tool', fromfile_prefix_chars='@',
                                     conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument('--weight_path', nargs='+', default=[],
                        help="weight file(s) to use for prediction")
    parser.add_argument('--image_path', nargs='+', default=[],
                        help='Image file(s) to use for prediction (type:*.nii.gz)')
    parser.add_argument('--select_dir', action='store_true',
                        help='if set, consider each element in ```image_path``` as directory and select all files in each ```image_path``` directory')
    parser.add_argument('--save_path', type=str, required=True, default='',
                        help='File save directory')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Threshold probability value to decide if a voxel is included in airway or not')
    parser.add_argument('--segmentation_only', action='store_true',
                        help='Do not label generation if set')
    # TODO add help
    parser.add_argument('--branch_penalty', type=float, default=16.)
    parser.add_argument('--prune_threshold', type=float, default=0.1)
    parser.add_argument('--use_bfs', action='store_true')
    parser.add_argument('--do_not_add_broken_parts', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')


    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    weight_path = []
    for ph in args.weight_path:
        if ph != '' and ph[0] != '#':
            weight_path.append(ph)
    
    image_path = []
    if not args.select_dir:
        for ph in args.image_path:
            if ph != ''and ph[0] != '#':
                image_path.append(ph.lstrip("./"))
    else:
        for dir in args.image_path:
            if dir != ''and dir[0] != '#':
                file_lists = sorted(os.listdir(dir))
                for ph in file_lists:
                    image_path.append(os.path.join(dir, ph).lstrip("./"))

    save_path = args.save_path
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except:
            pass

    # ---------- setting up models and result file ----------
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    threshold = args.threshold
    
    generation_info = pd.DataFrame()
    csv_path = save_path.rstrip('/').rstrip('\\') + '/' + "generation_info.csv"
    if os.path.exists(csv_path):
        generation_info = pd.read_csv(csv_path)

    trace_volume_by_gen_info = pd.DataFrame()
    csv_path = save_path.rstrip('/').rstrip('\\') + '/' + "trace_volume_by_gen_info.csv"
    if os.path.exists(csv_path):
        trace_volume_by_gen_info = pd.read_csv(csv_path)

    trace_slice_area_info = pd.DataFrame()
    csv_path = save_path.rstrip('/').rstrip('\\') + '/' + "trace_slice_area_info.csv"
    if os.path.exists(csv_path):
        trace_slice_area_info = pd.read_csv(csv_path)

    pixdim_info = pd.DataFrame()
    pixdim_csv_path = save_path.rstrip('/').rstrip('\\') + '/' + "pixdim_info.csv"
    if os.path.exists(pixdim_csv_path):
        pixdim_info = pd.read_csv(pixdim_csv_path)

    print(generation_info)
    models = [None for _ in range(len(weight_path))]

    for i, load_path in enumerate(weight_path):
        models[i]=SegAirwayModel(in_channels=1, out_channels=2)
        models[i].to(device)
        checkpoint = torch.load(load_path)
        models[i].load_state_dict(checkpoint['model_state_dict'])


    os.makedirs(save_path.rstrip('/').rstrip('\\') + '/extended_segment/', exist_ok=True)
    os.makedirs(save_path.rstrip('/').rstrip('\\') + '/orig_segment/', exist_ok=True)
    os.makedirs(save_path.rstrip('/').rstrip('\\') + '/extended_segment_before_postprocess/', exist_ok=True)
    os.makedirs(save_path.rstrip('/').rstrip('\\') + '/orig_segment_before_postprocess/', exist_ok=True)

    # ---------- start infer ----------
    for image_path in image_path:
        logging.log(logging.INFO, f"Start model infer of {image_path} ...")
        start_time = cur_time = time.time()
        # load and resize image if its scale is too different from train data
        raw_img = load_one_CT_img(image_path)
        orig_size = raw_img.shape
        if orig_size[0] < MIN_SLICE_COUNT:
            raw_img = transform.resize(raw_img.astype(float), (MIN_SLICE_COUNT, orig_size[1], orig_size[2])).astype(int)

        # make prob map and onehot segment
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
        logging.log(logging.INFO, f"Took {time.time() - cur_time:.3f}s for model infer")

        '''
        seg_processed,_ = post_process(seg_result_comb, threshold=threshold)
        seg_slice_label_I, connection_dict_of_seg_I, number_of_branch_I, tree_length_I = tree_detection(seg_processed, search_range=1,  branch_penalty=0.)
        seg_processed_II = add_broken_parts_to_the_result(connection_dict_of_seg_I, seg_result_comb, seg_processed, threshold = threshold,
                                                    search_range = 10, delta_threshold = 0.05, min_threshold = threshold * 0.6)
        '''
        logging.log(logging.INFO, "Start postprocessing...")
        time_start_sub = time.time()
        seg_processed_II = post_process_v2(seg_onehot_comb, seg_result_comb, threshold=threshold, min_threshold=0.6*threshold, add_broken_parts= not args.do_not_add_broken_parts)
        seg_processed_II = post_process_v2(seg_onehot_comb, seg_result_comb, threshold=threshold, min_threshold=0.6*threshold, add_broken_parts= not args.do_not_add_broken_parts)
        logging.log(logging.INFO, f"Took {time.time() - time_start_sub:.3f}s for postprocessing")

        print()
        _, cc_num_1 = label(seg_processed_II, return_num=True, connectivity=1)
        print("number of CC (connectivity=1):", cc_num_1)
        _, cc_num_2 = label(seg_processed_II, return_num=True, connectivity=2)
        print("number of CC (connectivity=2):", cc_num_2)
        _, cc_num_3 = label(seg_processed_II, return_num=True, connectivity=3)
        print("number of CC (connectivity=3):", cc_num_3)
        logging.log(logging.INFO, f"number of CC: ({cc_num_1}, {cc_num_2}, {cc_num_3})")


        seg_path_extended = (save_path.rstrip('/').rstrip('\\')
                        + '/extended_segment/'
                        + image_path[image_path.rfind('/') + 1:image_path.find('.')][image_path.rfind('\\') + 1:]
                        + "_segmentation.nii.gz")
        sitk.WriteImage(sitk.GetImageFromArray(seg_processed_II.astype(np.uint8)), seg_path_extended)
        seg_path_extended_before_postprocess = (save_path.rstrip('/').rstrip('\\')
                        + '/extended_segment_before_postprocess/'
                        + image_path[image_path.rfind('/') + 1:image_path.find('.')][image_path.rfind('\\') + 1:]
                        + "_segmentation.nii.gz")
        sitk.WriteImage(sitk.GetImageFromArray(seg_onehot_comb.astype(np.uint8)), seg_path_extended_before_postprocess)

        seg_processed_II_orig_size = transform.resize(seg_processed_II, orig_size, order=0, mode="edge", preserve_range=True, anti_aliasing=False)
        seg_onehot_comb_orig_size = transform.resize(seg_onehot_comb, orig_size, order=0, mode="edge", preserve_range=True, anti_aliasing=False)
        seg_path_orig = (save_path.rstrip('/').rstrip('\\')
                        + '/orig_segment/'
                        + image_path[image_path.rfind('/') + 1:image_path.find('.')][image_path.rfind('\\') + 1:]
                        + "_segmentation.nii.gz")
        sitk.WriteImage(sitk.GetImageFromArray(seg_processed_II_orig_size.astype(np.uint8)), seg_path_orig)
        seg_path_orig_before_postprocess = (save_path.rstrip('/').rstrip('\\')
                        + '/orig_segment_before_postprocess/'
                        + image_path[image_path.rfind('/') + 1:image_path.find('.')][image_path.rfind('\\') + 1:]
                        + "_segmentation.nii.gz")
        sitk.WriteImage(sitk.GetImageFromArray(seg_onehot_comb_orig_size.astype(np.uint8)), seg_path_orig_before_postprocess)

        has_pixdim = False
        pixdim = np.array([1., 1., 1.])
        try:
            img_header = nib.load(image_path).header
            for i in range(3):
                pixdim[i] = img_header['pixdim'][i + 1]
            has_pixdim = True
        except:
            pass
        
        cur_pixdim_info = {
            'path': seg_path_extended,
            'has_pixdim': has_pixdim,
            'pixdim_x': pixdim[0],
            'pixdim_y': pixdim[1],
            'pixdim_z': pixdim[2],
            'slice_count': orig_size[0]
        }

        pixdim_info = pd.DataFrame(pixdim_info.append(cur_pixdim_info, ignore_index=True))
        pixdim_info.to_csv(pixdim_csv_path, index=False)
        
        if not args.segmentation_only:
            logging.log(logging.INFO, f"Starting generation labeling...")
            cur_time = time.time()
            break_and_save(seg_path_extended, save_path, generation_info, trace_volume_by_gen_info, trace_slice_area_info, args, cur_pixdim_info)
            logging.log(logging.INFO, f"Took {time.time() - cur_time:3f}s for generation labeling")
        logging.log(logging.INFO, f"Total time elapsed: {time.time() - start_time:.3f}")
        logging.log(logging.INFO, '')

if __name__ == "__main__":
    main()