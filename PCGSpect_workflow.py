from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import ndimage
import librosa
import librosa.display
from scipy.io import wavfile
from PIL import Image
import streamlit as st


def pcg_spec_workflow(waveform_path):
    print("CHOSEN :", waveform_path)


    img_path = create_linearF_spec(waveform_path)
    spec_img=Image.open("./img-ui/spectrogram_sample.jpg")
    st.image(spec_img,caption='Generated Spectrogram')


    get_clusters(img_path)
    clus_spec_8 = Image.open("cluster_spec_img.jpg")
    st.image(clus_spec_8, caption='8 Clusters of Spectrogram Generated')




def create_linearF_spec(AUDIOFILE):

    #AUDIOFILE = "C:/MATLAB/physionet.org/files/challenge-2016/1.0.0/training-a/a0007.wav"
    print(AUDIOFILE)
    sample_rate, samples = wavfile.read(AUDIOFILE)
    samples = samples.astype(float)
    X = librosa.stft(samples)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sample_rate)
    #plt.colorbar()
    # if(L == "a"):
    #     v = "C:/Users/Uma Bala/OneDrive/Desktop/Sem7/Project 1/physionet spectro/Images/Training-B/Abnormal/" + ID + ".jpg"
        
    # elif(L == "n"):
    #     v = "C:/Users/Uma Bala/OneDrive/Desktop/Sem7/Project 1/physionet spectro/Images/Training-B/Normal/" + ID + ".jpg"

    v = "./img-ui/spectrogram_sample.jpg"
    plt.axis('off')
    plt.box(False)
    #import matplotlib.pyplot as plt
    plt.savefig(v, bbox_inches = "tight", transparent = True)
    with Image.open(v) as img:
        img.load()
    img = img.crop((10,10,1095,395))
    img.save(v)
    return v
    #img.show()    
    

def get_clusters(FILE_NAME):
    N = 8 # @ TEAM : Specify the number of clusters needed hereee 

    img = imread(FILE_NAME)/255
    #print("img size: ", img.shape)
    image_2D = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    
    kmeans = KMeans(n_clusters=N, random_state=0).fit(image_2D)
    clustered = kmeans.cluster_centers_[kmeans.labels_]
    clustered_3D = clustered.reshape(img.shape[0], img.shape[1], img.shape[2])
    # x = kmeans.labels_
    # unique, counts = np.unique(x, return_counts = True)
    #print("NUMBER OF CLUSTERS = ", N)
#     for x in range(0,N):
#         print("CLUSTER: ", unique[x], "-> ", counts[x])
    plt.figure(figsize=(14, 5))
    plt.imshow(clustered_3D)
    
    plt.axis('off')
    plt.box(False)
    #import matplotlib.pyplot as plt
    SAVE_FILE = "cluster_spec_img.jpg"
    plt.savefig(SAVE_FILE, bbox_inches = "tight", transparent = True)
    with Image.open(SAVE_FILE) as img:
        img.load()
    img = img.crop((10,10,1095,395))
    img.save(SAVE_FILE)
    print("FINISHED CLUSTERING STEP ROSSSSHNI")
   # img.show()
    #print(unique, counts)
