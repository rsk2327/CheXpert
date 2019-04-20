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

from utils import *

################# DEFINE PARAMETERS #################

disease = 'Pneumonia'
baseFolder  = "/home/santhosr/Documents/Chexpert"
modelID = 8
batchSize = 50



############### PREPROCESSING #######################

cols = ['Path',
 'Sex',
 'Age',
 'View',
 'AP/PA',
 'No_Finding',
 'Enlarged_Cardiomediastinum',
 'Cardiomegaly',
 'Lung_Opacity',
 'Lung_Lesion',
 'Edema',
 'Consolidation',
 'Pneumonia',
 'Atelectasis',
 'Pneumothorax',
 'Pleural_Effusion',
 'Pleural_Other',
 'Fracture',
 'Support_Devices']


trainFile = pd.read_csv(os.path.join(baseFolder,'train.csv'), names = cols, header=0)
validFile = pd.read_csv(os.path.join(baseFolder,'valid.csv'), names = cols, header=0)

trainFile["Path"] = trainFile.Path.apply(lambda x : x.replace('CheXpert-v1.0-small',"")[1:])
validFile["Path"] = validFile.Path.apply(lambda x : x.replace('CheXpert-v1.0-small',"")[1:])


selectCols = ['Path',"View",'Sex',"Pleural_Effusion", "Edema","Cardiomegaly","Pneumonia"]

trainFile = trainFile[selectCols]
validFile = validFile[selectCols]

# -1 for Uncertain, 0 for negative, 1 for positive

trainFile['isClean'] = trainFile.apply(lambda x : cleanLabel(x), axis = 1)
validFile['isClean'] = validFile.apply(lambda x : cleanLabel(x), axis = 1)

trainFile['train'] = False
validFile['train'] = True

trainFile = trainFile[trainFile.isClean==1]
validFile = validFile[validFile.isClean==1]


df = pd.concat([trainFile,validFile])

getLabel = partial(getLabel, disease = disease)
df['label'] = df.apply(lambda x : getLabel(x), axis = 1)

labelMap = {"Pleural_Effusion":0, "Edema":1,"Cardiomegaly":2,"Pneumonia":3,"Rest":4}





################# MODELING ##################################



getLabel = partial(getLabel, df = df, labelMap = labelMap)

print("Data Creation Start")
data = ImageItemList.from_df(df=df,path=baseFolder, cols='Path').split_from_df(col='train').label_from_func(getLabelDf).transform(get_transforms(),size=256).databunch(bs=batchSize).normalize()
print("Data Creation Complete")


learn = create_cnn(data, tmodels.resnet50, metrics=accuracy,pretrained=True)

# learn.load('/home/santhosr/Documents/Birad/ProcessedData/models/model_resnet50_acc668_loss600')

best_model_cb = partial(ModelTrackerCallback,id=modelID, modelName = "resnet50_"+disease)
learn.callback_fns.append(best_model_cb)




############ TRAINING #####################################
learn.unfreeze()
learn.fit(30,1e-5)
