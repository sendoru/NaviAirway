# needs dicom2nifti module
# conda install -c conda-forge dicom2nifti
# OR
# pip install dicom2nifti

# %%
USE_GUI_PROMPT = False

import os
import sys
import numpy as np
import SimpleITK as sitk
from PIL import Image
import pydicom
import cv2
import nibabel as nib
import dicom2nifti as d2n
import dicom2nifti.settings as d2n_settings
import gc
import shutil
import logging

if USE_GUI_PROMPT:
    from PyQt5.QtCore import Qt, QObject
    import tkinter as tk
    from tkinter.filedialog import askopenfilename, asksaveasfilename, askdirectory
    import PyQt5.QtWidgets as qtw
    import PyQt5.QtGui as qtg
    from PyQt5.QtCore import Qt, QObject

from skimage import transform
import matplotlib.pyplot as plt


# %%
dicom_dir_path = "./data/Airway/distortion"
save_path = "./data/Airway/distortion_3D_d2n"

# %%
if USE_GUI_PROMPT:
    tk.Tk().withdraw()
    logging.basicConfig(level=logging.CRITICAL)
    app = qtw.QApplication(sys.argv)

    msg = qtw.QMessageBox()
    msg.setIcon(qtw.QMessageBox.Information)
    msg.setText("Select DICOM image source directory.")
    msg.setWindowTitle("")
    msg.setStandardButtons(qtw.QMessageBox.Ok)
    msg.exec_()

    dicom_dir_path = ""

    while True:
        dicom_dir_path = askdirectory()
        if dicom_dir_path != "":
            break
        msg = qtw.QMessageBox()
        msg.setIcon(qtw.QMessageBox.Warning)
        msg.setText("Invaild path. Try again.")
        msg.setWindowTitle("")
        msg.setStandardButtons(qtw.QMessageBox.Ok)
        msg.exec_()

    msg = qtw.QMessageBox()
    msg.setIcon(qtw.QMessageBox.Information)
    msg.setText("Select save directory.")
    msg.setWindowTitle("")
    msg.setStandardButtons(qtw.QMessageBox.Ok)
    msg.exec_()

    save_path = ""

    while True:
        save_path = askdirectory()
        if save_path != "":
            break
        msg = qtw.QMessageBox()
        msg.setIcon(qtw.QMessageBox.Warning)
        msg.setText("Invaild path. Try again.")
        msg.setWindowTitle("")
        msg.setStandardButtons(qtw.QMessageBox.Ok)
        msg.exec_()

# %%
if not USE_GUI_PROMPT:
    dicom_dir_path = ""
    while True:
        dicom_dir_path = input("Enter DICOM directory path: ")
        if os.path.isdir(dicom_dir_path):
            break
        else:
            print("Invalid path. Try again.")

    save_path = ""
    while True:
        save_path = input("Enter save directory path: ")
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, os.path.split(dicom_dir_path)[-1] + "_nifti")
            os.makedirs(save_path, exist_ok=True)
            break
        else:
            print("Invalid path. Try again.")

# %%
test_case_names = []
for disease_name in os.listdir(dicom_dir_path):
    disease_path = os.path.join(dicom_dir_path, disease_name)
    if os.path.isdir(disease_path):
        for patient_name in os.listdir(disease_path):
            patient_path = os.path.join(disease_path, patient_name)
            if os.path.isdir(patient_name):
                for date in os.listdir(patient_path):
                    test_case_dir = os.path.join(patient_path, date)
                    if os.path.isdir(test_case_dir):
                        test_case_names.append(os.path.join(disease_name, patient_name, date))

test_case_names.sort()
if '.DS_Store' in test_case_names:
    test_case_names.remove('.DS_Store')

# print(test_case_names)

# %%
os.makedirs(save_path, exist_ok=True)
temp_save_path = os.path.join(save_path, "temp")
os.makedirs(temp_save_path, exist_ok=True)
for fname in os.listdir(temp_save_path):
    os.remove(os.path.join(temp_save_path, fname))

# %%
for i, test_case_name in enumerate(test_case_names):
    test_case_dir = os.path.join(dicom_dir_path, test_case_name)

    d2n.convert_directory(test_case_dir, temp_save_path)
    print("\x1B[H\x1B[J")
    fnames = sorted(os.listdir(temp_save_path))

    print(f"processing {test_case_name} ({i + 1}/{len(test_case_names)})...")

    if len(fnames) == 0:
        print("No valid DICOM image serieses detected. Skipping current folder...")
        
    elif len(fnames) == 1:
        print("Detected single DICOM image series. Converting to NiFTi...")
        source_path = os.path.join(temp_save_path, fnames[0])
        dest_path = os.path.join(save_path, test_case_name.replace(os.path.sep, '_'))
        shutil.copy(source_path, dest_path)

    else:
        print("More than one DICOM image serieses detected. Select the file(s) to save: ")
        print("-" * 40)
        print("0) Save all serieses")
        print()
        for j, fname in enumerate(fnames):
            nib_obj = nib.load(os.path.join(temp_save_path, fname))
            print(f"{j + 1}) Series name: {'.'.join(fname.split('.')[:-2])}", f"   Image dimension: {nib_obj.header['dim'][1:4]}", sep='\n')
        print("-" * 40)
        choice = -1
        while choice == -1:
            raw_input = input(f"Input your choice and press ENTER ({0}~{len(fnames)}): ")
            if not raw_input.isnumeric() or int(raw_input) < 0 or int(raw_input) > len(fnames):
                print("Invaid value. Please try again.")
                continue
            choice = int(raw_input)

        if choice == 0:
            for j, fname in enumerate(fnames):
                source_path = os.path.join(temp_save_path, fname)
                dest_path = os.path.join(save_path, test_case_name.replace(os.path.sep, '_') + '_' + fname)
                shutil.copy(source_path, dest_path)
        else:
            source_path = os.path.join(temp_save_path, fnames[choice - 1])
            dest_path = os.path.join(save_path, test_case_name.replace(os.path.sep, '_'))
            shutil.copy(source_path, dest_path)

    if len(fnames) != 0:
        print("Saved file. ")
    for fname in os.listdir(temp_save_path):
        os.remove(os.path.join(temp_save_path, fname))

    gc.collect()
    print()


