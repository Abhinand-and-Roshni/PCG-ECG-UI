#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import pandas as pd
# names = ['file_name', 'label']
# df = pd.read_csv('C:/MATLAB/physionet.org/files/challenge-2016/1.0.0/training-a/REFERENCE.csv', names = names )


# In[2]:


# df


# In[ ]:


import os
import sys
import numpy as np
import pandas as pd


# In[22]:


# Import local Libraries
sys.path.insert(0, os.path.dirname(os.getcwd()))
from features.feature_extractor import Features


# In[5]:


waveform_path = "C:/Users/Roshni/Downloads/A03035.mat"
print(waveform_path)


# In[23]:


# Instantiate
labels = [1,-1]
fs = 300
ecg_features = Features(file_path=waveform_path, fs=fs, feature_groups=['full_waveform_features'])

# Calculate ECG features
ecg_features.extract_features(
    filter_bandwidth=[3, 45], n_signals=None, show=True, 
    labels=labels, normalize=True, polarity_check=True,
    template_before=0.25, template_after=0.4
)


# In[24]:


from features import feature_extractor
dic_fwff = feature_extractor.features
# features = ecg_features.get_features()
# features
dic_fwff


# In[9]:


# features.to_csv('features_ecg.csv')


# In[10]:


waveform_path


# In[25]:


fs = 300
rri_features = Features(file_path=waveform_path, fs=fs, feature_groups=['rri_features'])
print(waveform_path)
# Calculate ECG features
rri_features.extract_features(
    filter_bandwidth=[3, 45], n_signals=None, show=True, 
    labels=labels, normalize=True, polarity_check=True,
    template_before=0.25, template_after=0.4
)


# In[26]:


from features import feature_extractor
dic_rri = feature_extractor.features


# In[27]:


dic_rri


# In[28]:


dic1 = [dic_rri]


# In[29]:


df11 = pd.DataFrame(dic1)
df11 = df11.iloc[:,1:]
df11


# In[30]:


df11_list = df11.values


# In[31]:


from sklearn.decomposition import PCA
import pickle as pkl

pca_1 = pkl.load(open("pca16_ecg_2017.pkl",'rb'))
df111 = pca_1.transform(df11)
df111


# In[32]:


fin_df = np.concatenate((df11, df111),1)
fin_df


# In[34]:


pickled_model = pkl.load(open("RFC_Boosting_ECG17.pkl", "rb"))

pickled_model.predict(fin_df)


# In[36]:


# fin_df


# In[53]:


# df  = pd.read_csv("C:/Users/Roshni/Downloads/tttt.csv", names = ['c','d'])


# In[54]:


# df1 = df.iloc[:,0]
# df


# In[55]:


# df['c'] = df['c'].str[4:]


# In[57]:


# df.to_csv('C:/Users/Roshni/Downloads/tttt.csv')


# In[104]:


# x = "C:/Users/Roshni/Downloads/A03001.csv"
# x = x.split(".")[0].split("/")[4]
# x


# In[105]:


# df_actual = pd.read_csv("C:/Users/Roshni/OneDrive/Desktop/Sem7/Project 1/ECG UI/2016_17_ALL_ECG_SUBJECTS_WITH_LABEL.csv", index_col = False)
# #x = waveform_path
#     # MODIFY [5] BASED ON THE FUNCTION FILE_SELECTOR'S FOLDER PATH !!
# x = x.split(".")[0]
# print(x)
# y = df_actual.loc[df_actual['file_name'] == x]['label']



# z = df_actual[df_actual['file_name'].str.contains(x)]
# m = z['label']
# m = np.array(m)
# if(m=="N"):
#     print("fin")
# else:
#     print("nfin")


# In[76]:


# df_actual['file_name']


# In[ ]:




