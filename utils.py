import pandas as pd
import numpy as np

import os
from PIL import Image

import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.torch_core import *
from fastai.callback import *
from fastai.basic_train import *
import torchvision.models as tmodels



def cleanLabel(x):
    
    labelCount = 0
    
    if x.Pleural_Effusion == 1:
        labelCount += 1
        
    if x.Edema == 1:
        labelCount += 1
    
    if x.Cardiomegaly ==1:
        labelCount += 1
    
    if x.Pneumonia == 1:
        labelCount += 1
    
    return labelCount



def getLabelDf(x,df, labelMap):
    
#     print(x)
    
    x = x[36:]          #To account for the extra "././" added before the Path variable
#     print(x)
    x = df.loc[df.Path == x] 
#     print(x)
    
    return labelMap[x.label.values[0]]
    
    

def getLabel(x,disease):
    
    if x[disease] ==1:
        return disease
    else:
        return "Rest"


class ModelTrackerCallback(TrackerCallback):
    "A `TrackerCallback` that saves the model when monitored quantity is best."
    def __init__(self, learn:Learner, path:str='/home/santhosr/Documents/Courses/CIS700/Project/models',id:int=None,monitor:str='val_loss', mode:str='auto',modelName:str='resnet50'):
        super().__init__(learn, monitor=monitor, mode=mode)
        
        self.bestAcc = 0.0001
        self.folderPath = path
        self.id = id
        self.modelName = modelName
        super().__post_init__()

    def on_epoch_end(self, epoch, **kwargs:Any)->None:
        "Compare the value monitored to its best score and maybe save the model."

        acc = float(self.learn.recorder.metrics[epoch-1][0])
        val_loss = self.learn.recorder.val_losses[epoch-1]

        if acc>self.bestAcc:
            self.bestAcc = acc
            if self.id==None:
                fileName = 'model_'+self.modelName+'_acc'+str(int(acc*1000))+"_loss"+str(int(val_loss*1000))
            else:
                fileName = 'model_'+self.modelName+'_id' + str(self.id) + '_acc' + str(int(acc*1000)) + "_loss" + str(int(val_loss*1000))
            fileName = os.path.join(self.folderPath, fileName)
            self.learn.save(fileName)
