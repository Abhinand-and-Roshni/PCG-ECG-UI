from features.feature_extractor import Features
from features import feature_extractor
from features import imagefeatures
import streamlit as st
import pandas as pd
import re
import os
import sys
import numpy as np
from sklearn.decomposition import PCA
import pickle as pkl
from preprocessing import specConvert
from PIL import Image
from ECGworkflow import ecg_workflow
from PCGworkflow import pcg_workflow
from PCGSpect_workflow import pcg_spec_workflow
from PCGSpec_Prediction import prediction_8_cluster

# test msg updating 4th dec

def file_selector(folder_path='./TestData/'):
    filenames = os.listdir(folder_path)
    #filenames = filenames.sort()
    print(filenames)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


# Import local Libraries
sys.path.insert(0, os.path.dirname(os.getcwd()))

st.title('PCG CLASSIFICATION - BEST MODEL')
with st.form(key='my_form1'):
    waveform_path = None
    try:
        waveform_path = file_selector()
        if(waveform_path.split('.')[1] == 'mat' or waveform_path.split('.')[1] == 'MAT'):
            st.error('Enter a WAV file!')
        if(len(waveform_path) != 0):
            st.write('You selected `%s`' % waveform_path)
            st.success('File successfully loaded!')
        else:
            raise TypeError
    except TypeError:
        st.error('No file entered')
    submit_button = st.form_submit_button(label='Predict!')
    try:
        pcg_workflow(waveform_path)
    except EOFError:
        st.error('Please enter WAV file!')


st.title("ECG CLASSIFICATION - BEST MODEL")
with st.form(key='my_form'):
    #   waveform_path = st.file_uploader("Upload ECG file", type=["mat"])
    waveform_path = None
    try:
        waveform_path = file_selector()
        if(len(waveform_path) != 0):
            st.write('You selected `%s`' % waveform_path)
            st.success('File successfully loaded!', icon="✅")
        else:
            raise TypeError
            #raise MatReadError("Mat file appears to be empty")
    except TypeError:
        print('No file Entered!')

    submit_button = st.form_submit_button(label='Predict!')
    try:
        ecg_workflow(waveform_path)
    except TypeError:
        st.error('Please enter MAT File!')



st.title("PCG Classification - Spectrogram Clustering")
with st.form(key='my_form2'):
    waveform_path = None
    try:
        waveform_path = file_selector()
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


    #waveform_path = "C:/Users/Roshni/Downloads/A03035.mat"
    # print(waveform_path)
