from features.feature_extractor import Features
from scipy.io.matlab.miobase import (MatFileReader, docfiller, matdims, read_dtype,
                                     arr_to_chars, arr_dtype_number, MatWriteError,
                                     MatReadError, MatReadWarning)
from features import feature_extractor
from features import imagefeatures
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import Sequential, optimizers
from keras.layers import Input, Dense, Dropout, LSTM, RepeatVector, TimeDistributed, BatchNormalization
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from auto_encoder import encoderdecoder
import streamlit as st
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import load_model
import pandas as pd
import re
from scipy.io import loadmat
import numpy as np
import sys
from sklearn.decomposition import PCA
import pickle as pkl
from preprocessing import specConvert
from PIL import Image
from ECGworkflow import ecg_workflow
from PCGworkflow import pcg_workflow
from PCGSpect_workflow import pcg_spec_workflow
from PCGSpec_Prediction import prediction_8_cluster
import os
from rev2ECGWorkflow import r2workflow_lstmae
from rev1ECGWorkflow import r1ecg_lr_vanillaAE

def file_selector(folder_path='./TestData/'):
    filenames = os.listdir(folder_path)
    #filenames = filenames.sort()
    print(filenames)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

waveform_path = file_selector()

# Import local Libraries
sys.path.insert(0, os.path.dirname(os.getcwd()))
tab1, tab2 = st.tabs(["Phase I", "Phase II"])
#============= PHASE 2 wORFLOW STARTS =================================================================================================
with tab2:
    try:
        
#==================================== REVIEW 2 STARTS HERE================================================
        st.title("ECG Classification using LSTM-AE with PCA")
        print("35:",waveform_path.split(".")[2])
        if(waveform_path.split(".")[2]== 'wav'):
            waveform_path1 = waveform_path.replace('wav', 'mat')
            print("**38:",waveform_path1)
            if(os.path.exists(waveform_path1) == True):
                #waveform_path = waveform_path1
                print("96: ",waveform_path)
            else:
                st.error('No corresponding MAT file found for this WAV file!')
        else:
            waveform_path1 = waveform_path
        with st.form(key='my_form_p2_r2'):
            #st.header("PHASE II")
            st.write('You selected `%s`' % waveform_path1)
            st.success('File successfully loaded!', icon="✅")
            submit_button = st.form_submit_button(label='Predict!')
            r2workflow_lstmae(waveform_path1)




        #*****************REVIEW 1*****************************************************************
        st.title("ECG Classification using Autoencoders and LR-Bayes Method")
        print("35:",waveform_path.split(".")[2])
        if(waveform_path.split(".")[2]== 'wav'):
            waveform_path1 = waveform_path.replace('wav', 'mat')
            print("**38:",waveform_path1)
            if(os.path.exists(waveform_path1) == True):
                #waveform_path = waveform_path1
                print("96: ",waveform_path)
            else:
                st.error('No corresponding MAT file found for this WAV file!')
        else:
            waveform_path1 = waveform_path
        with st.form(key='my_form_p2_r1'):
            #st.header("PHASE II")
            st.write('You selected `%s`' % waveform_path1)
            st.success('File successfully loaded!', icon="✅")
            submit_button = st.form_submit_button(label='Predict!')
            r1ecg_lr_vanillaAE(waveform_path1)


    except MatReadError:
        st.error("Provide suitable file.")


    



#===============PHASE 1 WORKFLOW=================================================================================================
with tab1:
    #st.header("PHASE I (July 2022 - December 2022)")
        
    st.title("PCG Classification - Spectrogram Clustering")
    with st.form(key='my_form2'):
        #waveform_path = None
        try:
            #waveform_path = file_selector()
            if(len(waveform_path) != 0):
                st.write('You selected `%s`' % waveform_path)
                st.success('File successfully loaded!', icon="✅")
            else:
                raise TypeError
            
        except TypeError:
            print('No file Entered!')

        submit_button = st.form_submit_button(label='Predict!')
        try:
            pcg_spec_workflow(waveform_path)

            # path = "./img-ui/spectrogram_sample.jpg" HAS THE IMAGE OF SPECTROGRAM - BEFORE CLUSTER
            # path = "cluster_spec_img.jpg" HAS THE IMAGE OF 8 CLUSTERED K MEANS SPECTROGRAM 

            pred_path = "cluster_spec_img.jpg"
            prediction_8_cluster(pred_path, waveform_path)



        except ValueError:
                st.error('Please enter WAV File!')

    st.title('PCG CLASSIFICATION - BEST MODEL')
    with st.form(key='my_form1'):
    # waveform_path = None
        try:
        #  waveform_path = file_selector()
            if(waveform_path.split('.')[2] == 'mat' or waveform_path.split('.')[2] == 'MAT'):
                print("68:",waveform_path.split('.')[2])
                st.error('Enter a WAV file!')
            if(len(waveform_path) != 0):
                st.write('You selected `%s`' % waveform_path)
                st.success('File successfully loaded!')
            else:
                raise TypeError
        except TypeError:
            st.error('No file entered')
        submit_button = st.form_submit_button(label='Predict!')

        if(waveform_path.split('.')[2] not in ' mat'): # or waveform_path.split('.')[2] != ' MAT'):
            try:
                pcg_workflow(waveform_path)
            except EOFError:
                st.error("Enter WAV File!")
        # except EOFError:
        #     st.error('Please enter WAV file!')
        




    st.title("ECG CLASSIFICATION - BEST MODEL")
    print("35:",waveform_path.split(".")[2])
    if(waveform_path.split(".")[2]== 'wav'):
        waveform_path1 = waveform_path.replace('wav', 'mat')
        print("**38:",waveform_path1)
        if(os.path.exists(waveform_path1) == True):
            waveform_path = waveform_path1
            print("96: ",waveform_path)
        else:
            st.error('No corresponding MAT file found for this WAV file!')
    print("99: ", waveform_path)
    with st.form(key='my_form'):
        #   waveform_path = st.file_uploader("Upload ECG file", type=["mat"])
        #waveform_path = None
        try:
            #waveform_path = file_selector()
            if(len(waveform_path) != 0):
                st.write('You selected `%s`' % waveform_path)
                st.success('File successfully loaded!', icon="✅")
            else:
                raise TypeError
                #raise MatReadError("Mat file appears to be empty")
        except TypeError:
            print('No file Entered!')
        except ValueError:
            print("Value Error - MAT FILE")

        submit_button = st.form_submit_button(label='Predict!')
        try:
            ecg_workflow(waveform_path)
        except TypeError:
            st.error('Please enter MAT File!')
        except ValueError:
            st.error('Entered file should be of MAT type!')






        #waveform_path = "C:/Users/Roshni/Downloads/A03035.mat"
        # print(waveform_path)
