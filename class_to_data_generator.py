from tensorflow.keras.utils import Sequence
from io import BytesIO
import pandas as pd
import numpy as np

class ImageDataGenerator(Sequence):
  def __init__(self , df_data , batch_size=32 , shuffle=True ,target_size=None):
    
    assert shuffle in [False, True] ,"this must be True or False only"

    super().__init__() #important to work well
    
    self.dataset    = df_data
    self.batch_size = batch_size
    self.shuffle    = shuffle
    self.target_size=target_size
    #use data and get indexes to use in getitem by fit of keras 
    self.indexes = np.arange(len(self.dataset))

    #call this at begining of each instance 
    self.on_epoch_end()
    
  def __len__(self):  #this to calculate num of batches per epoch 
    return int(np.ceil(len(self.dataset)  / self.batch_size ))

  def __getitem__(self,idx): #this what happen to each batch asked by model (0,1,2,3)
    start = idx * self.batch_size  
    end   = min((idx + 1) * self.batch_size, len(self.indexes))  #t his hadel the last batch 
    batch_indexes = self.indexes[start:end] 

    img=[]
    lab=[]

    for i in batch_indexes:
      image = Image.open(BytesIO(self.dataset.iloc[i,0]['bytes']))
      label = Image.open(BytesIO(self.dataset.iloc[i,1]['bytes']))

      if self.target_size:
        image=image.resize(self.target_size)
        label=label.resize(self.target_size)

      image = np.array(image).astype(np.float32) / 255.0
      label = np.array(label).astype(np.float32) / 255.0

      img.append(image)
      lab.append(label)
    
    img= np.array(img)
    lab= np.array(lab)

    return img, lab

  def on_epoch_end(self):
        """Shuffle at epoch end"""
        if self.shuffle:
            np.random.shuffle(self.indexes)
