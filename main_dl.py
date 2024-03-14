#csv input 
import pandas as pd 
data = pd.read_csv('TyphoonKR2013~2014data.csv')
data = data.dropna()
ydata = data['rad'].values
xdata = []
for i, rows in data.iterrows():
    xdata.append([rows['cp'],rows['ms'],rows['kmh']])

import tensorflow as tf
import numpy as np 
from tkinter import*
    
#neural network
model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='swish'),
        tf.keras.layers.Dense(128, activation='swish'),
        tf.keras.layers.Dense(256, activation='swish'),
        tf.keras.layers.Dense(1, activation='swish'),  
        
    ])
model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])

#learning    
model.fit(np.array(xdata),np.array(ydata),epochs=1000)
#output
Prediction = model.predict([[990,24,86]])
print(Prediction)






