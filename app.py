from features.feature_extractor import Features
from scipy.io.matlab.miobase import (MatFileReader, docfiller, matdims, read_dtype,
                                     arr_to_chars, arr_dtype_number, MatWriteError,
                                     MatReadError, MatReadWarning)
from features import feature_extractor
from features import imagefeatures

from auto_encoder import encoderdecoder
import streamlit as st
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, losses
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
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

# test msg updating 4th dec

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
#============= PHASE 2 wORFLOW =================================================================================================
with tab2:
    try:
        st.title("ECG Classification using Autoencoders and LR-Bayes Method")
        with st.form(key='my_form_p2_r1'):
            #st.header("PHASE II")
            st.write('You selected `%s`' % waveform_path)
            st.success('File successfully loaded!', icon="✅")
            submit_button = st.form_submit_button(label='Predict!')
            def read_mat_files(file_path):
                df_list = []
                mat = loadmat(file_path)
                print(mat['val'][0] , ' | file: ' , file_path)
                x = file_path.split('.')[0]
                data = mat['val'][0]
                data = np.append(x, data)
                df_list.append(data)
                df = pd.DataFrame(df_list)
                return df
            df_list = read_mat_files(waveform_path)
            df_list = df_list.drop(0, axis = 1)

            st.write("Reading the ECG File:")
            st.dataframe(df_list)
            df_list = df_list.astype(float)
            wave_tf = tf.convert_to_tensor(df_list)
            wave_tf = tf.cast(wave_tf, dtype=tf.float32)

            # Recreate the model
            model = encoderdecoder.detector()

            model.compile(optimizer='adam', loss='mae')
            model.build(input_shape=(None, 3600))
            model.load_weights("./model_folder/autoencoder_weights.h5")
            x = model.encoder(wave_tf)
            print(x)
            xx = pd.DataFrame(x.numpy())

            st.write("Newly Constructed Features using Encoder:")
            st.dataframe(xx)

            with open( "./model_folder/LR_Bayes_16.pkl", "rb" ) as f:
                LRB = pkl.load(f)
            y = LRB.predict(xx)
            st.text("-- PREDICTED RESULT FOR FILE " + waveform_path + " -- ")
            print(y)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            if(y==-1):
                st.write("SUBJECT ECG ABNORMAL")
            elif(y==1):
                st.write("SUBJECT ECG NORMAL")

            st.text("-- ACTUAL RESULT FOR FILE " + waveform_path + " -- ")
            df_actual = pd.read_csv("2016_17_ALL_ECG_SUBJECTS_WITH_LABEL.csv")
            x = waveform_path
            print("TESTTTTTTTTT"+waveform_path)
            # MODIFY [5] BASED ON THE FUNCTION FILE_SELECTOR'S FOLDER PATH !!
            x = x.split(".")[1].split(".")[0].split("/")[-1]
            print("X IS ")
            print(x)
            x = str(x)
            z = df_actual[df_actual['file_name'].str.contains(x)]
            #print(df_actual['file_name']+str("IS THE PATIENT!"))
            m = z['label']
            m = np.array(m)
            #print("*****87:", m)
            if(m == [1]):
                print("patient normal")
                st.text("SUBJECT ECG NORMAL")
            elif(m == [-1]):
                print("patient abnormal")
                st.text("SUBJECT ECG ABNORMAL")
            else:
                # ground truth not provided
                st.text("Ground Truth Inconclusive Result (~)")
            
        







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
