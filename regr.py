import shutil
import os
import time
from datetime import datetime
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchsample.transforms import RandomRotate, RandomTranslate, RandomFlip, ToTensor, Compose, RandomAffine
from torchvision import transforms
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from dataloader.py import MRDataset
import model

from sklearn import metrics

def extract_predictions(task, plane, train=True):
    assert task in ['acl', 'meniscus', 'abnormal']
    assert plane in ['axial', 'coronal', 'sagittal']
    
    models = os.listdir('models/')

    model_name = list(filter(lambda name: task in name and plane in name, models))[0]
    model_path = f'models/{model_name}'

    mrnet = torch.load(model_path)
    _ = mrnet.eval()
    
    train_dataset = MRdataset('data/', 
                              task, 
                              plane, 
                              transform=None, 
                              train=train, 
                              normalize=False)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=1, 
                                               shuffle=False, 
                                               num_workers=10, 
                                               drop_last=False)
    predictions = []
    labels = []
    with torch.no_grad():
        for image, label, _ in tqdm_notebook(train_loader):
            logit = mrnet(image.cuda())
            prediction = torch.sigmoid(logit)
            predictions.append(prediction.item())
            labels.append(label.item())

    return predictions, labels

import os
task = 'acl'
results = {}

for plane in ['axial', 'coronal', 'sagittal']:
    predictions, labels = extract_predictions(task, plane)
    results['labels'] = labels
    results[plane] = predictions
    
X = np.zeros((len(predictions), 3))
X[:, 0] = results['axial']
X[:, 1] = results['coronal']
X[:, 2] = results['sagittal']

y = np.array(labels)

logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X, y)

task = 'acl'
results_val = {}

for plane in ['axial', 'coronal', 'sagittal']:
    predictions, labels = extract_predictions(task, plane, train=False)
    results_val['labels'] = labels
    results_val[plane] = predictions

y_pred = logreg.predict_proba(X_val)[:, 1]
metrics.roc_auc_score(y_val, y_pred)