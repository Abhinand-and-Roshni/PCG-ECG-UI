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


#hi roshni can u see this ???

def file_selector(folder_path='C:/Users/Roshni/OneDrive/Desktop/Sem7/Project 1/ECG UI/TestData/'):
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


#waveform_path = "C:/Users/Roshni/Downloads/A03035.mat"
# print(waveform_path)
