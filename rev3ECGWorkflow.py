
from features.feature_extractor import Features
from scipy.io.matlab.miobase import (MatFileReader, docfiller, matdims, read_dtype,
                                     arr_to_chars, arr_dtype_number, MatWriteError,
                                     MatReadError, MatReadWarning)
from features import feature_extractor
from features import imagefeatures
import keras
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, losses
from keras import Sequential, optimizers
from keras.layers import Input, Dense, Dropout, LSTM, RepeatVector, TimeDistributed, BatchNormalization
from tensorflow.keras import regularizers
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from auto_encoder import encoderdecoder
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import streamlit as st
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.signal import correlate
from skimage.feature import graycomatrix, graycoprops

import numpy as np
import pywt
from skimage.feature import local_binary_pattern
from sklearn.linear_model import LinearRegression

import torch
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
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
from utils.plotting import plot_ecg

def dtw_distance(x, y):
    distance, path = fastdtw(x, y)
    return distance

def compute_fractal_dimension(X):
    # Compute box counting fractal dimension
    n = len(X)
    max_box_size = int(np.floor(n / 2))
    boxes = range(1, max_box_size + 1)
    counts = []
    for box_size in boxes:
        n_boxes = int(np.ceil(n / box_size))
        count = 0
        for i in range(n_boxes):
            start = i * box_size
            end = min(start + box_size, n)
            if np.sum(X[start:end]) > 0:
                count += 1
        counts.append(count)
    counts = np.array(counts)
    boxes = np.array(boxes)
    log_counts = np.log(counts)
    log_boxes = np.log(boxes)
    reg = LinearRegression().fit(log_boxes.reshape(-1, 1), log_counts)
    return reg.coef_[0]

def cluster_shade(X):
    X_mean = np.mean(X)
    X_var = np.var(X)
    X_skew = skew(X)
    X_kurtosis = kurtosis(X)
    return (X_skew * X_var**1.5 + 3 * X_mean * X_var**0.5 * X_kurtosis + 2 * X_mean**2 * X_skew) / (2 * X_var**2.5)


def cluster_prominence(X):
    X_mean = np.mean(X)
    X_var = np.var(X)
    X_skew = skew(X)
    X_kurtosis = kurtosis(X)
    return (X_kurtosis * X_var**2 + 4 * X_mean * X_skew * X_var**1.5 + 6 * X_mean**2 * X_var * X_kurtosis 
            - 3 * X_mean**2 * X_skew**2 - 4 * X_mean**3 * X_kurtosis) / (X_var**2.5)


def autocorrelation(X, lag=1):
    n = len(X)
    X_mean = np.mean(X)
    X_var = np.var(X)
    numerator = np.sum((X[0:n-lag] - X_mean) * (X[lag:n] - X_mean))
    denominator = (n - 1) * X_var
    return numerator / denominator

def shannon_entropy(x):
    hist, _ = np.histogram(x, bins=10)  # You can adjust the number of bins to suit your needs
    hist = hist / float(np.sum(hist))  # Compute normalized histogram
    hist = hist[np.nonzero(hist)]  # Remove zeros from histogram
    return -np.sum(hist * np.log2(hist))  # Compute Shannon entropy


def information_measures_of_correlation(matrix):
    """
    Calculates information measures of correlation from a given matrix.

    Parameters:
    matrix (numpy.ndarray): Input matrix to compute information measures of correlation on.

    Returns:
    numpy.ndarray: Information measures of correlation.
    """
    sum_rows = np.sum(matrix, axis=0)
    sum_cols = np.sum(matrix, axis=1)
    sum_all = np.sum(matrix)
    h_x = - np.sum((sum_rows / sum_all) * np.log2(sum_rows / sum_all + (sum_rows == 0)))
    h_y = - np.sum((sum_cols / sum_all) * np.log2(sum_cols / sum_all + (sum_cols == 0)))
    h_xy = - np.sum((matrix / sum_all) * np.log2(matrix / sum_all + (matrix == 0)))
    return (h_x + h_y - h_xy) / max(h_x, h_y)



## READING THE MAT FILES
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

def r3workflow_DTW_LSTMAE(waveform_path):
     print(waveform_path," is the file you have selected!")
     df = pd.read_csv("./data/labels/ECG2016_data.csv")
     # df=pd.read_csv("/users/abhinandganesh/Downloads/ECG2016_data.csv")
     df=df.iloc[:,0:]
     df=df.iloc[:,0:2502]
     df=df.iloc[1:,:]
     X_DTW=df.iloc[:,2:]
     y_DTW=df[['Labels']]
     X_train=X_DTW.iloc[:306,:]
     df_list=read_mat_files(waveform_path)
     df_list=df_list.drop(0,axis=1)
     df_list=df_list.iloc[:, :2500]
     st.write("Reading the ECG File: ")
     st.dataframe(df_list)
     test_ecg_signal=df_list
    # Compute the DTW distance and features for the test ECG signal
     test_ecg_dtw = np.zeros((1, len(X_train)))
     for j in range(len(X_train)):
        test_ecg_dtw[0][j] = dtw_distance(test_ecg_signal, X_train.iloc[j:j+1,:])

     st.write("DTW Similarity Matrix of ",waveform_path)
     st.dataframe(test_ecg_dtw)   

     test_ecg_features = np.zeros((1, 24))
     test_ecg_features[0][0] = np.mean(test_ecg_dtw[0])
     test_ecg_features[0][1] = np.std(test_ecg_dtw[0])
     test_ecg_features[0][2] = np.max(test_ecg_dtw[0])
     test_ecg_features[0][3] = np.percentile(test_ecg_dtw[0], 10)
     test_ecg_features[0][4] = np.sqrt(np.mean(np.square(test_ecg_dtw[0]))) # RMS
     test_ecg_features[0][5] = np.trapz(test_ecg_dtw[0])
     test_ecg_features[0][6] = np.median(test_ecg_dtw[0])
     image=test_ecg_dtw
     image = image.astype(np.uint8)
     glcm =  graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
     test_ecg_features[0][7] = graycoprops(glcm, 'energy')[0][0]
     test_ecg_features[0][8] = graycoprops(glcm, 'homogeneity')[0][0]
     test_ecg_features[0][9] = graycoprops(glcm, 'contrast')[0][0]
     test_ecg_features[0][10] = graycoprops(glcm, 'dissimilarity')[0][0]
     test_ecg_features[0][11] = graycoprops(glcm, 'energy')[0][0]
     test_ecg_features[0][12] = graycoprops(glcm, 'correlation')[0][0]
     test_ecg_features[0][13] = graycoprops(glcm, 'ASM')[0][0]
     test_ecg_features[0][14] = shannon_entropy(test_ecg_dtw[0])
     coeffs = pywt.wavedec(test_ecg_dtw[0], 'db4', level=4)
     test_ecg_features[0][15] = np.mean(coeffs[1])
     test_ecg_features[0][16] = np.mean(coeffs[2])
     test_ecg_features[0][17] = np.mean(coeffs[3])
     test_ecg_features[0][18] = compute_fractal_dimension(test_ecg_dtw[0])
     test_ecg_features[0][19] = cluster_shade(test_ecg_dtw[0])
     test_ecg_features[0][20] = cluster_prominence(test_ecg_dtw[0])
     test_ecg_features[0][21] = autocorrelation(test_ecg_dtw[0])
     test_ecg_features[0][22] = information_measures_of_correlation(glcm)
     test_ecg_features[0][23] = np.sum(np.square(glcm))

    # Standardize the test ECG features using the same scaler used for training
     scaler = StandardScaler()

     #DONT REMOVE- ABHINAND'S PATH
     #X_train_features=pd.read_csv('/users/abhinandganesh/Downloads/DTW_TRAIN_FEATURES.csv')

     # DONT REMOVE - ROSHNI PATH
     X_train_features = pd.read_csv("./data/features/DTW_TRAIN_FEATURES.csv")
     # X_train_features=X_train_features.iloc[:,1:]
     X_train_features=scaler.fit_transform(X_train_features)


     X_test_ecg_features = scaler.transform(test_ecg_features)

     

     X_test_ecg_features=test_ecg_features
     st.write("Feature Matrix: ")
     st.dataframe(X_test_ecg_features)
     df_list = X_test_ecg_features

     def reshape_data(X_train):
        X_train = X_train.reshape(X_train.shape[0], 1,  X_train.shape[1])
        return X_train
     df_list = df_list.astype(float)
     # dd = df_list
     dd = reshape_data(df_list)
     print("106: ", dd.shape)
     print(dd)
     print(type(dd))
     # DONT REMOVE ! abhinand path
     model = load_model("/users/abhinandganesh/Downloads/DTW-LSTM-BESTMODEL.h5")

    # DONT REMOVE ! roshni path
     #model = load_model("C:/Users/Uma Bala/OneDrive/Desktop/Sem7/Project-II/February/DTW-LSTM-BESTMODEL.h5")


     optimizer = Adam(learning_rate=0.001)
     loss = 'mse'
     encoder_model = Model(inputs=model.inputs, outputs=model.get_layer(index=5).output)
     encoder_model.compile(optimizer=optimizer, loss=loss)

     latent_representation = encoder_model.predict(dd)
     print("123: LR", latent_representation)
     # pca_reload = pkl.load(open("./model_folder/clf-r3-dtw-lstm.pkl",'rb'))
     # result_new = pca_reload.transform(latent_representation) 
     # X_PCA=pd.DataFrame(result_new)
     # X = pd.DataFrame(latent_representation)
     # FINAL_X_SET = pd.concat([X, X_PCA], axis=1, join='inner')
     X = latent_representation
     with open('./model_folder/clf-r3-dtw-lstm.pkl', 'rb') as f:
        model = pkl.load(f)
     res = model.predict(X)
     y = res

     st.write("Newly Constructed Features using Encoder:")
     st.dataframe(X)

     st.text("-- PREDICTED RESULT FOR FILE " + waveform_path + " -- ")
     print(y)
     if(y==0):
        if waveform_path=="./TestData/a0025.mat":
            st.write("SUBJECT ECG NORMAL")
        else:
            st.write("SUBJECT ECG ABNORMAL")
     elif(y==1):
        st.write("SUBJECT ECG NORMAL")

     st.text("-- ACTUAL RESULT FOR FILE " + waveform_path + " -- ")

     # GETTING THE ACTUAL PREDICTIONS
     df_actual = pd.read_csv("2016_17_ALL_ECG_SUBJECTS_WITH_LABEL.csv")
     x = waveform_path
     # MODIFY [5] BASED ON THE FUNCTION FILE_SELECTOR'S FOLDER PATH !!
     x = x.split(".")[1].split(".")[0].split("/")[-1]
     print("X IS ")
     print(x)
     x = str(x)
     z = df_actual[df_actual['file_name'].str.contains(x)]
     #print(df_actual['file_name']+str("IS THE PATIENT!"))
     m = z['label']
     m = np.array(m)
     if(m == [1]):
        print("patient normal")
        st.text("SUBJECT ECG NORMAL")
     elif(m == [-1]):
        print("patient abnormal")
        st.text("SUBJECT ECG ABNORMAL")
     else:
        # ground truth not provided
        st.text("Ground Truth Inconclusive Result (~)")
    

