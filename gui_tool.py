import os
import sys
from time import sleep
import numpy as np
from scipy.ndimage import zoom
import torch
import copy
import pandas as pd
import SimpleITK as sitk
from PIL import Image
import pydicom
import cv2
import nibabel as nib
import os
import skimage.io as io
import PyQt5.QtWidgets as qtw
import PyQt5.QtGui as qtg
from PyQt5.QtCore import Qt, QObject
import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename, askdirectory

from func.points_to_mesh import generate_obj, produce_3d_obj
tk.Tk().withdraw() # part of the import if you are not using other tkinter 

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

# select file -> make prob. map 이건 파일 선택할 때 하나로 합치고
# Load Model Weights, Load Raw Image, Save
# 참고: train에 안 쓰인거 : EXACT09 - 1, 5, 6, 18

'''
TODO
앙상블 예측 구현 (ㅅㅂ!!!!)
스켈레톤 모델 저장
GUI에 사진 띄우는 거 미리 다 계산해두고 띄워두는 식으로 바꾸기
레이어 순서대로 재생, 역재생, 일시정지, 1칸씩
UI 좀 고치기
    좀 긴 로딩 때 뜨는 창 제대로 띄우기 (prob. map 생성, postprocessing)
    status bar에 뭔가 띄워두기 (현재 띄워둔 레이어, 모델 로딩 여부 등)
'''

class NaviAirWayGUI(qtw.QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = SegAirwayModel(in_channels=1, out_channels=2)
        self.model.to(self.device)
        self.model_2 = SegAirwayModel(in_channels=1, out_channels=2)
        self.model_2.to(self.device)
        self.image = None
        self.label = None
        self.prob_map = None
        self.seg_onehot_bp = None
        self.seg_onehot_ap = None
        self.threshold = 0.7
        self.layer_idx = 0

        self.initUI()

    def initUI(self):
        self.img_label_left = qtw.QLabel('img_left', self)

        # make statusbar
        self.statusBar()

        # make menubar
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        statusbar = self.statusBar()
        # statusbar.

        # menuBar actions
        select_model_action = qtw.QAction('Load Weights for Model 1...', self)
        select_model_action.triggered.connect(lambda: self.select_model_weights(self.model))

        select_model_2_action = qtw.QAction('Load Weights for Model 2...', self)
        select_model_2_action.triggered.connect(lambda: self.select_model_weights(self.model_2))

        select_image_action = qtw.QAction('Load Raw Image...', self)
        select_image_action.triggered.connect(self.select_image)
        select_image_action.triggered.connect(self.render_image)
        select_image_action.triggered.connect(self.adjust_slider_range)

        select_label_action = qtw.QAction('Load Original Label...', self)
        select_label_action.triggered.connect(self.select_label)

        save_voxel_action_bp = qtw.QAction('Save Voxel as *.obj (Before Postprocessing)...', self)
        save_voxel_action_bp.triggered.connect(self.save_voxel_bp_obj)

        save_voxel_action_ap = qtw.QAction('Save Voxel as *.obj (After Postprocessing)...', self)
        save_voxel_action_ap.triggered.connect(self.save_voxel_ap_obj)

        save_voxel_action_ap_niigz = qtw.QAction('Save Voxel as *.nii.gz (After Postprocessing)...', self)
        save_voxel_action_ap_niigz.triggered.connect(self.save_voxel_ap_gz)

        save_voxel_action_ap_3_niigz = qtw.QAction('Save Voxel as *.nii.gz (After Postprocessing, Comparison with Orig. label)...', self)
        save_voxel_action_ap_3_niigz.triggered.connect(self.save_voxel_ap_3_gz)

        save_skeleton_action = qtw.QAction('Save Skeleton...', self)
        save_skeleton_action.triggered.connect(self.save_skeleton)

        exit_action = qtw.QAction('Exit', self)
        exit_action.triggered.connect(qtw.qApp.quit)

        self.make_prob_map_action = qtw.QAction('Make Probability Map', self)
        self.make_prob_map_action.triggered.connect(self.make_prob_map)
        self.make_prob_map_action.triggered.connect(self.render_pred)

        self.set_threshold_action = qtw.QAction('Make Segmentation Map / Set Threshold', self)
        self.set_threshold_action.triggered.connect(self.set_threshold)
        self.set_threshold_action.triggered.connect(self.make_seg_onehot)
        self.set_threshold_action.triggered.connect(self.render_pred)

        toogle_prediction_action_group = qtw.QActionGroup(self)

        self.toggle_prediction_view_prob = qtw.QAction('Probability Map', toogle_prediction_action_group, checkable=True)
        self.toggle_prediction_view_prob.triggered.connect(self.render_pred)

        self.toggle_prediction_view_seg_bp = qtw.QAction('Segmentaion (Before Postprocessing)', toogle_prediction_action_group, checkable=True)
        self.toggle_prediction_view_seg_bp.triggered.connect(self.render_pred)

        self.toggle_prediction_view_seg_ap = qtw.QAction('Segmentaion (After Postprocessing)', toogle_prediction_action_group, checkable=True)
        self.toggle_prediction_view_seg_ap.setChecked(True)
        self.toggle_prediction_view_seg_ap.triggered.connect(self.render_pred)

        self.toggle_compare_with_label = qtw.QAction('Compare With Label', toogle_prediction_action_group, checkable=True)
        self.toggle_compare_with_label.setDisabled(True)
        self.toggle_prediction_view_seg_bp.triggered.connect(self.render_pred)

        self.set_layer_idx_action = qtw.QAction('Set Layer Index', self)
        self.set_layer_idx_action.triggered.connect(self.set_layer_idx_manual)
        self.set_layer_idx_action.triggered.connect(self.render_image)
        self.set_layer_idx_action.triggered.connect(self.render_pred)

        self.status_model_loaded_action = qtw.QAction('Model Loaded?', self, checkable=True)
        self.status_model_loaded_action.setDisabled(True)
        self.status_image_loaded_action = qtw.QAction('Image Loaded?', self, checkable=True)
        self.status_image_loaded_action.setDisabled(True)
        self.status_label_loaded_action = qtw.QAction('Label Loaded?', self, checkable=True)
        self.status_label_loaded_action.setDisabled(True)
        self.status_prob_loaded_action = qtw.QAction('Prob. Map Done?', self, checkable=True)
        self.status_prob_loaded_action.setDisabled(True)
        self.status_seg_loaded_action = qtw.QAction('Segmentation Done?', self, checkable=True)
        self.status_seg_loaded_action.setDisabled(True)


        # add menuBar action to menuBar
        filemenu = menubar.addMenu('&File')
        filemenu.addAction(select_model_action)
        filemenu.addAction(select_model_2_action)
        filemenu.addAction(select_image_action)
        filemenu.addAction(select_label_action)
        filemenu.addSeparator()
        filemenu.addAction(save_voxel_action_bp)
        filemenu.addAction(save_voxel_action_ap)
        filemenu.addAction(save_voxel_action_ap_niigz)
        filemenu.addAction(save_voxel_action_ap_3_niigz)
        filemenu.addAction(save_skeleton_action)
        filemenu.addSeparator()
        filemenu.addAction(exit_action)

        editmenu = menubar.addMenu('&Edit')
        editmenu.addAction(self.make_prob_map_action)
        editmenu.addAction(self.set_threshold_action)

        viewmenu = menubar.addMenu('&View')
        # toggle_prediction_view_action = viewmenu.addMenu('Toggle Prediction View')
        viewmenu.addAction(self.toggle_prediction_view_prob)
        viewmenu.addAction(self.toggle_prediction_view_seg_bp)
        viewmenu.addAction(self.toggle_prediction_view_seg_ap)
        viewmenu.addSeparator()
        viewmenu.addAction(self.toggle_compare_with_label)
        viewmenu.addSeparator()
        viewmenu.addAction(self.set_layer_idx_action)

        statusmenu = menubar.addMenu('&Status')
        statusmenu.addAction(self.status_model_loaded_action)
        statusmenu.addAction(self.status_image_loaded_action)
        statusmenu.addAction(self.status_prob_loaded_action)
        statusmenu.addAction(self.status_seg_loaded_action)


        # add labels for image
        self.label_left = qtw.QLabel('No Image', self)
        self.label_left.move(0, 0 + menubar.size().height())
        self.label_right = qtw.QLabel('No Image', self)
        self.label_right.move(512, 0 + menubar.size().height())

        self.label_left.resize(512, 512)
        self.label_right.resize(512, 512)

        self.label_left.setAlignment(Qt.AlignCenter)
        self.label_right.setAlignment(Qt.AlignCenter)

        font_left = self.label_left.font()
        font_left.setPointSize(20)
        self.label_left.setFont(font_left)

        font_right = self.label_right.font()
        font_right.setPointSize(20)
        self.label_right.setFont(font_right)

        # 옆에 있는 (현재 레이어) / (레이어 갯수) 누르면 maunally하게 숫자 쳐서 넣을수 있게도 구현
        # + 재생, 역재생, 정지 구현 (이건 선택사항에 가깝긴 한데)
        self.layer_idx_slider = qtw.QSlider(Qt.Horizontal, self)
        self.layer_idx_slider.setValue(1)
        self.layer_idx_slider.setRange(1, 1)
        self.layer_idx_slider.valueChanged.connect(self.set_layer_idx_slider)
        self.layer_idx_slider.valueChanged.connect(self.render_image)
        self.layer_idx_slider.valueChanged.connect(self.render_pred)
        self.layer_idx_slider.setFixedSize(784, 32)
        self.layer_idx_slider.move(144, 512 + menubar.size().height() + 16)

        self.layer_idx_label = qtw.QLabel(f"{self.layer_idx + 1}/{self.layer_idx_slider.maximum()}", self)
        self.layer_idx_label.setAlignment(Qt.AlignRight)
        self.layer_idx_label.move(900, 512 + menubar.size().height() + 32 - self.layer_idx_label.fontInfo().pixelSize() // 2)
        
        self.setWindowTitle("NaviAirway")
        self.setFixedSize(1024, 512 + menubar.size().height() + 64 + statusbar.size().height())
        self.show()


    def select_model_weights(self, model):
        try:
            fn = askopenfilename(defaultextension='*.pkl', title='Select Model Weights File...', filetypes=[('*.pkl files', '*.pkl'), ('All files', '*.*')])
            if fn == '':
                return
            ckpoint = torch.load(fn)
            model.load_state_dict(ckpoint['model_state_dict'])
            self.status_model_loaded_action.setChecked(True)
            return True
        except:
            msg = qtw.QMessageBox()
            msg.setIcon(qtw.QMessageBox.Critical)
            msg.setText("Cannot load weights")
            msg.setInformativeText('The file might be corrupted or you selected wrong file.')
            msg.setWindowTitle("Error")
            msg.exec_()
            return False

    def select_image(self):
        try:
            fn = askopenfilename(defaultextension='*.gz', title='Select Image File...', filetypes=[('*.gz files', '*.gz'), ('All files', '*.*')])
            if fn == '':
                return
            self.image = load_one_CT_img(fn)
            self.layer_idx = 0
            self.status_image_loaded_action.setChecked(True)
            self.label = None
            self.prob_map = None
            self.seg_onehot_bp = None
            self.seg_onehot_ap = None
            self.status_label_loaded_action.setChecked(False)
            self.status_prob_loaded_action.setChecked(False)
            self.status_seg_loaded_action.setChecked(False)
            self.toggle_compare_with_label.setChecked(False)
            self.toggle_compare_with_label.setDisabled(True)
            return True
        except:
            msg = qtw.QMessageBox()
            msg.setIcon(qtw.QMessageBox.Critical)
            msg.setText("Cannot load image")
            msg.setInformativeText('The file might be corrupted or you selected wrong file.')
            msg.setWindowTitle("Error")
            msg.exec_()
            return False
        
    def select_label(self):
        if type(self.image) == type(None):
            msg = qtw.QMessageBox()
            msg.setIcon(qtw.QMessageBox.Warning)
            msg.setText("Image is not loaded.")
            msg.setInformativeText('Please load image and try again.')
            msg.setWindowTitle("Error")
            msg.exec_()
            return False
        try:
            fn = askopenfilename(defaultextension='*.gz', title='Select Label File...', filetypes=[('*.gz files', '*.gz'), ('All files', '*.*')])
            if fn == '':
                return
            label = io.imread(fn, plugin='simpleitk')
            label = np.array(label, dtype=float)
            if label.shape != self.image.shape:
                msg = qtw.QMessageBox()
                msg.setIcon(qtw.QMessageBox.Warning)
                msg.setText("Label shape is not mached to image.")
                msg.setInformativeText('.')
                msg.setWindowTitle("Error")
                msg.exec_()
                return False
            self.label = label
            
            self.status_label_loaded_action.setChecked(True)
            self.toggle_compare_with_label.setDisabled(False)
            return True
        except:
            msg = qtw.QMessageBox()
            msg.setIcon(qtw.QMessageBox.Critical)
            msg.setText("Cannot load label")
            msg.setInformativeText('The file might be corrupted or you selected wrong file.')
            msg.setWindowTitle("Error")
            msg.exec_()
            return False
        
    def make_prob_map(self):
        if type(self.image) == type(None):
            msg = qtw.QMessageBox()
            msg.setIcon(qtw.QMessageBox.Warning)
            msg.setText("Image is not loaded.")
            msg.setInformativeText('Please load image and try again.')
            msg.setWindowTitle("Error")
            msg.exec_()
            return False
        elif not self.status_model_loaded_action.isChecked():
            msg = qtw.QMessageBox()
            msg.setIcon(qtw.QMessageBox.Warning)
            msg.setText("Model weights is not loaded.")
            msg.setInformativeText('Please load model weights and try again.')
            msg.setWindowTitle("Error")
            msg.exec_()
            return False

        # TODO 창에 진행 퍼센트 표기하기
        # progess bar 같은게 제일 낫겠지만
        # 그거라도 안되면 semantic_segment_crop_and_cat에서 출력되는거 후킹해오거나
        # 일단 비동기 처리나 잘 구현하자
        msg = qtw.QMessageBox(self)
        msg.setIcon(qtw.QMessageBox.Question)
        msg.setText("Select model(s) for prediction.")
        msg.setInformativeText("Press 'Yes' to use both model,\nor press 'No' to use only model 1.")
        msg.setWindowTitle("Select Model")

        msg.setStandardButtons(qtw.QMessageBox.Yes | qtw.QMessageBox.No | qtw.QMessageBox.Cancel)
        res_dual_model = msg.exec_()

        if res_dual_model == qtw.QMessageBox.Cancel:
            return False

        msg = qtw.QMessageBox(self)
        msg.setIcon(qtw.QMessageBox.Question)
        msg.setText("Use reduced image for prediction?")
        msg.setWindowTitle("Title")

        msg.setStandardButtons(qtw.QMessageBox.Yes | qtw.QMessageBox.No | qtw.QMessageBox.Cancel)
        res_reduced_image = msg.exec_()

        if res_reduced_image == qtw.QMessageBox.Cancel:
            return False

        msg = qtw.QMessageBox(self)
        msg.setIcon(qtw.QMessageBox.Information)
        msg.setText("Making prob map...")
        msg.open()
        sleep(.10)

        try:
            if res_reduced_image == qtw.QMessageBox.Yes:
                image = zoom(self.image, (0.25, 1, 1))
            elif res_reduced_image == qtw.QMessageBox.No:
                image = self.image
            if res_dual_model == qtw.QMessageBox.Yes:
                prob_map_1 = semantic_segment_crop_and_cat(image, self.model, self.device, crop_cube_size=[32, 128, 128], stride=[16, 64, 64],)
                prob_map_2 = semantic_segment_crop_and_cat(image, self.model_2, self.device, crop_cube_size=[32, 128, 128], stride=[16, 64, 64],)
                self.prob_map = (prob_map_1 + prob_map_2) / 2
            elif res_dual_model == qtw.QMessageBox.No:
                self.prob_map = semantic_segment_crop_and_cat(image, self.model, self.device, crop_cube_size=[32, 128, 128], stride=[16, 64, 64],)

            if res_reduced_image == qtw.QMessageBox.Yes:
                zeros = np.zeros(self.image.shape, dtype=self.prob_map.dtype)
                self.prob_map = zoom(self.prob_map, (self.image.shape[0] / self.prob_map.shape[0], 1, 1))
                if zeros.shape[0] >= self.prob_map.shape[0]:
                    zeros[:self.prob_map.shape[0],:,:] = self.prob_map
                else:
                    zeros = self.prob_map[:zeros.shape[0],:,:]
                self.prob_map = zeros

            msg.close()
            return True
        
        except:
            msg.close()
            return False

    def select_image_and_make_prob_map(self):
        res1 = self.select_image()
        res2 = self.make_prob_map()
        return res1 and res2

    def adjust_slider_range(self):
        if type(self.image) != type(None):
            self.layer_idx_slider.setRange(1, self.image.shape[0])
            self.layer_idx_slider.setValue(1)
            self.layer_idx_label.setText(f"{self.layer_idx + 1}/{self.layer_idx_slider.maximum()}")

    def make_seg_onehot(self):
        if type(self.image) == type(None) or type(self.prob_map) == type(None):
            return False
            
        msg = qtw.QMessageBox(self)
        msg.setIcon(qtw.QMessageBox.Information)
        msg.setText("Making segmentation map...")
        msg.open()
        self.seg_onehot_bp = np.array(self.prob_map>self.threshold, dtype=np.int)
        seg_processed, _ = post_process(self.seg_onehot_bp, threshold=self.threshold)
        seg_slice_label_I, connection_dict_of_seg_I, number_of_branch_I, tree_length_I = tree_detection(seg_processed, search_range=2)
        seg_processed_II = add_broken_parts_to_the_result(connection_dict_of_seg_I, self.prob_map, seg_processed, threshold = self.threshold,
                                                  search_range = 10, delta_threshold = 0.05, min_threshold = self.threshold * (4/7))
        self.seg_onehot_ap = seg_processed_II
        msg.close()
        return True
    
    def render_image(self):
        if type(self.image) == type(None):
            self.label_left.clear()
            self.label_left.setText("No Image")
            return
        cmap_gray = plt.get_cmap('gray')

        img = self.image[self.layer_idx,:,:]
        img_rgb = (cmap_gray((img - img.min()) / (img.max() - img.min())) * (2**16 - 1)).astype(np.uint16)[:,:,:3]
        img_rgb = (img_rgb / (2**8)).astype(np.uint8)
        img_left = qtg.QImage(img_rgb, img_rgb.shape[1], img_rgb.shape[0], 3 * img_rgb.shape[1], qtg.QImage.Format_RGB888)
        self.label_left.setPixmap(qtg.QPixmap(img_left))
        self.update()

    def render_pred(self):
        # TODO 원본 라벨과 비교 기능 넣기
        if self.toggle_prediction_view_prob.isChecked():
            pred_all = self.prob_map
        elif self.toggle_prediction_view_seg_bp.isChecked():
            pred_all = self.seg_onehot_bp
        else:
            pred_all = self.seg_onehot_ap
        
        if type(pred_all) == type(None):
            self.label_right.clear()
            self.label_right.setText("No Image")
            return 

        cmap_viridis = plt.get_cmap('viridis')
        cmap_Greens = plt.get_cmap('Greens')

        if self.toggle_prediction_view_prob.isChecked():
            pred = self.prob_map[self.layer_idx,:,:]
            pred_rgb = (cmap_viridis(pred) * (2**16 - 1)).astype(np.uint16)[:,:,:3]

        elif self.toggle_prediction_view_seg_bp.isChecked():
            pred = self.seg_onehot_bp[self.layer_idx,:,:]
            pred_rgb = (cmap_Greens(pred.astype(float)) * (2**16 - 1)).astype(np.uint16)[:,:,:3]
        else:
            pred = self.seg_onehot_ap[self.layer_idx,:,:]
            pred_rgb = (cmap_Greens(pred.astype(float)) * (2**16 - 1)).astype(np.uint16)[:,:,:3]
        pred_rgb = (pred_rgb / (2**8)).astype(np.uint8)

        if self.toggle_compare_with_label.isChecked() and type(self.label) != type(None):
            cur_label = self.label[self.layer_idx,:,:]
            tp = self.label * pred
            fp = (1 - self.label) * pred
            fn = self.label * (1 - pred)
            cmap_Blues = plt.get_cmap('Blues')      # for fn
            cmap_Reds = plt.get_cmap('Reds')        # for fp
            tp_rgb = (cmap_Greens(tp.astype(float)) * (2**16 - 1)).astype(np.uint16)[:,:,:3]
            fp_rgb = (cmap_Greens(fp.astype(float)) * (2**16 - 1)).astype(np.uint16)[:,:,:3]
            fn_rgb = (cmap_Greens(fn.astype(float)) * (2**16 - 1)).astype(np.uint16)[:,:,:3]
            pred_rgb = np.minimum(np.minimum(tp_rgb, fp_rgb), fn_rgb)
            pred_rgb = (pred_rgb / (2**8)).astype(np.uint8)
        
        img_right = qtg.QImage(pred_rgb, pred_rgb.shape[1], pred_rgb.shape[0], 3 * pred_rgb.shape[1], qtg.QImage.Format_RGB888)
        self.label_right.setPixmap(qtg.QPixmap(img_right))
        self.update()


    # def render_image_and_pred(self):

    #     if type(self.image) == type(None):
    #         msg = qtw.QMessageBox()
    #         msg.setIcon(qtw.QMessageBox.Critical)
    #         msg.setText("Image is not loaded")
    #         msg.setInformativeText('Please load image and try again.')
    #         msg.setWindowTitle("Error")
    #         msg.exec_()
    #         return False
        
    #     cmap_gray = plt.get_cmap('gray')
    #     cmap_viridis = plt.get_cmap('viridis')
    #     cmap_Blues = plt.get_cmap('Blues')

    #     img = self.image[self.layer_idx,:,:]
    #     img_rgb = (cmap_gray((img - img.min()) / (img.max() - img.min())) * (2**16 - 1)).astype(np.uint16)[:,:,:3]
    #     # img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    #     if self.toggle_prediction_view_prob.isChecked():
    #         pred = self.prob_map[self.layer_idx,:,:]
    #         pred_rgb = (cmap_viridis(pred) * (2**16 - 1)).astype(np.uint16)[:,:,:3]

    #     elif self.toggle_prediction_view_seg_bp.isChecked():
    #         pred = self.seg_onehot_bp[self.layer_idx,:,:]
    #         pred_rgb = (cmap_Blues(pred.astype(float)) * (2**16 - 1)).astype(np.uint16)[:,:,:3]
    #     else:
    #         pred = self.seg_onehot_ap[self.layer_idx,:,:]
    #         pred_rgb = (cmap_Blues(pred.astype(float)) * (2**16 - 1)).astype(np.uint16)[:,:,:3]

    #     # pred_rgb = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)

    #     img_rgb = (img_rgb / (2**8)).astype(np.uint8)
    #     pred_rgb = (pred_rgb / (2**8)).astype(np.uint8)
    #     # img_rgb = cv2.resize(img_rgb, (512, 512))
    #     # pred_rgb = cv2.resize(img_rgb, (512, 512))

    #     img_left = qtg.QImage(img_rgb, img_rgb.shape[1], img_rgb.shape[0], 3 * img_rgb.shape[1], qtg.QImage.Format_RGB888)
    #     img_right = qtg.QImage(pred_rgb, pred_rgb.shape[1], pred_rgb.shape[0], 3 * pred_rgb.shape[1], qtg.QImage.Format_RGB888)

    #     # self.pixmap_left.loadFromData(img_left)
    #     # self.pixmap_right.fromImage(img_right)
    #     self.label_left.setPixmap(qtg.QPixmap(img_left))
    #     self.label_right.setPixmap(qtg.QPixmap(img_right))
    #     self.label_left.resize(512, 512)
    #     self.label_right.resize(512, 512)

    #     self.update()


    def set_threshold(self):
        if type(self.image) == type(None):
            msg = qtw.QMessageBox()
            msg.setIcon(qtw.QMessageBox.Warning)
            msg.setText("Image is not loaded.")
            msg.setInformativeText('Please load image and try again.')
            msg.setWindowTitle("Error")
            msg.exec_()
            return False
        
        if type(self.prob_map) == type(None):
            msg = qtw.QMessageBox(self)
            msg.setIcon(qtw.QMessageBox.Question)
            msg.setText("Probability map is not ready.")
            msg.setInformativeText('You need probability map to make segmentation result. Do you want to make it first?')
            msg.setStandardButtons(qtw.QMessageBox.Yes | qtw.QMessageBox.No)
            msg.setWindowTitle("")
            res = msg.exec_()
            if res == qtw.QMessageBox.Yes:
                self.make_prob_map()
            else:
                return False
        while True:
            threshold, ok = qtw.QInputDialog.getDouble(self, 'Input', 'Input segmentation threshold (0.0 ~ 1.0, deault=0.7)', value=self.threshold, min=0.0, max=1.0)
            if threshold < 0 or threshold > 1:
                msg = qtw.QMessageBox()
                msg.setIcon(qtw.QMessageBox.Critical)
                msg.setText("Invalid Value")
                msg.setInformativeText('Try again.')
                msg.setWindowTitle("Error")
                msg.exec_()
            elif not ok:
                return
            else:
                break
        self.threshold = threshold
        return True
        
    def set_layer_idx_manual(self):
        if type(self.image) == type(None):
            msg = qtw.QMessageBox()
            msg.setIcon(qtw.QMessageBox.Warning)
            msg.setText("Image is not loaded.")
            msg.setInformativeText('Please load image and try again.')
            msg.setWindowTitle("Error")
            msg.exec_()
            return False
        
        while True:
            idx, ok = qtw.QInputDialog.getInt(self, 'Input', f'Input layer index (1 ~ {self.image.shape[0]})', value=self.layer_idx+1, min=1, max=self.image.shape[0])
            if not ok or idx < 1 or idx > self.image.shape[0]:
                msg = qtw.QMessageBox()
                msg.setIcon(qtw.QMessageBox.Critical)
                msg.setText("Invalid Value")
                msg.setInformativeText('Try again.')
                msg.setWindowTitle("Error")
                msg.exec_()
            else:
                break
        self.layer_idx_slider.setValue(idx)
        self.layer_idx = idx - 1
        return True
    
    def set_layer_idx_slider(self):
        self.layer_idx = self.layer_idx_slider.value() - 1
        self.layer_idx_label.setText(f"{self.layer_idx + 1}/{self.layer_idx_slider.maximum()}")
        return True
        
    def save_voxel_bp_obj(self):
        if type(self.seg_onehot_bp) == type(None):
            msg = qtw.QMessageBox()
            msg.setIcon(qtw.QMessageBox.Warning)
            msg.setText("Segmentation map is is not prepared.")
            msg.setInformativeText('Please make segmentaion map and try again.')
            msg.setWindowTitle("Error")
            msg.exec_()
            return False
        
        filename = asksaveasfilename(defaultextension='*.obj', title='Save Voxel...', filetypes=[('*.obj files', '*.obj'), ('All files', '*.*')])
        generate_obj(filename, [], self.seg_onehot_bp)
        return True

    def save_voxel_ap_obj(self):
        if type(self.seg_onehot_bp) == type(None):
            msg = qtw.QMessageBox()
            msg.setIcon(qtw.QMessageBox.Warning)
            msg.setText("Segmentation map is is not prepared.")
            msg.setInformativeText('Please make segmentaion map and try again.')
            msg.setWindowTitle("Error")
            msg.exec_()
            return False
        
        filename = asksaveasfilename(defaultextension='*.obj', title='Save Voxel...', filetypes=[('*.obj files', '*.obj'), ('All files', '*.*')])
        generate_obj(filename, [], self.seg_onehot_ap)
        return True
    
    def save_voxel_ap_gz(self):
        if type(self.seg_onehot_bp) == type(None):
            msg = qtw.QMessageBox()
            msg.setIcon(qtw.QMessageBox.Warning)
            msg.setText("Segmentation map is is not prepared.")
            msg.setInformativeText('Please make segmentaion map and try again.')
            msg.setWindowTitle("Error")
            msg.exec_()
            return False
        
        filename = asksaveasfilename(defaultextension='*.nii.gz', title='Save Voxel...', filetypes=[('*.nii.gz files', '*.nii.gz'), ('All files', '*.*')])
        sitk.WriteImage(sitk.GetImageFromArray(self.seg_onehot_ap), filename)
        return True

    def save_voxel_ap_3_gz(self):
        if type(self.seg_onehot_bp) == type(None) or type(self.label) == type(None):
            msg = qtw.QMessageBox()
            msg.setIcon(qtw.QMessageBox.Warning)
            msg.setText("Segmentation map or gt label is not prepared.")
            msg.setInformativeText('Please make segmentaion map and try again.')
            msg.setWindowTitle("Error")
            msg.exec_()
            return False
        
        dirname = askdirectory(title='Save Voxel...')
        if dirname == '':
            return False
        tp = self.label * self.seg_onehot_ap
        fp = (1 - self.label) * self.seg_onehot_ap
        fn = self.label * (1 - self.seg_onehot_ap)
        sitk.WriteImage(sitk.GetImageFromArray(tp), dirname + '/' + dirname.split('/')[-1] + "_tp.nii.gz")
        sitk.WriteImage(sitk.GetImageFromArray(fp), dirname + '/' + dirname.split('/')[-1] + "_fp.nii.gz")
        sitk.WriteImage(sitk.GetImageFromArray(fn), dirname + '/' + dirname.split('/')[-1] + "_fn.nii.gz")
        sitk.WriteImage(sitk.GetImageFromArray(self.seg_onehot_ap), dirname + '/' + dirname.split('/')[-1] + "_pred.nii.gz")
        return True

    def save_skeleton(self):
        msg = qtw.QMessageBox()
        msg.setIcon(qtw.QMessageBox.Information)
        msg.setText("Not Implemented")
        msg.setInformativeText('Sorry.')
        msg.setWindowTitle("Error")
        msg.exec_()

if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    ex = NaviAirWayGUI()
    app.exec_()