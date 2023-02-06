import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import layers, losses
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model

class detector(Model):
    def __init__(self):
        super(detector, self).__init__()
        self.encoder = tf.keras.Sequential([
                                            layers.Dense(32, activation='relu'),
                                            layers.Dense(16, activation='relu'),
                                            layers.Dense(8, activation='relu')
        ])
        self.decoder = tf.keras.Sequential([
                                            layers.Dense(16, activation='relu'),
                                            layers.Dense(32, activation='relu'),
                                            layers.Dense(3600, activation='sigmoid')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded