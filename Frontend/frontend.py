import streamlit as st
import gc
import os
import time
import json
import torch
import requests
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from matplotlib import pyplot as plt
# from cddd.inference import InferenceModel
from PIL import Image, ImageOps, ImageEnhance
from pytorch_lightning import LightningModule as LM

gc.collect()

# Options

opt = {
    'image_size': 224,
    'is_grayscale': False,
    'image_parsed': False,
    'dataset_size': 50000,
    'train': True,
    'epochs': 30,
    'load': False
}

# PATHS

ABSOLUTE_PATH = '/home/reshyurem/Documents/SMAI/'
IMAGE_PATH = ABSOLUTE_PATH + 'train_images/'
IMAGE_TENSOR_FILE = IMAGE_PATH + 'image_tensor.pt'
CDDD_PATH = ABSOLUTE_PATH + 'data/embeddings.csv'
MODEL_FILE = ABSOLUTE_PATH + '60000_images_60_epochs.pt'

# Helper Functions

def load_image(path):
    # Opening image
    im = Image.open(path).convert('L' if opt['is_grayscale'] else 'RGB')

    ratio = float(224) / max(im.size)
    ns = tuple([int(x * ratio) for x in im.size])
    im = im.resize(ns, Image.BICUBIC)
    ni = Image.new("L", (224, 224), "white")
    ni.paste(im, ((224 - ns[0]) // 2,
                        (224 - ns[1]) // 2))
    ni = ImageOps.expand(ni, int(np.random.randint(5, 25, size=1)), "white")
    im = ni
  
    # Enhancing image
    im = ImageEnhance.Contrast(ImageOps.autocontrast(im)).enhance(2.0)
    # Contrast adjustment
    im = ImageOps.autocontrast(im)
    im = im.resize((opt['image_size'],opt['image_size']))
    im = transforms.ToTensor()(im)
    return im
    
# Initialize a Cuda device if available
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Define the model

class Img2CDDD(LM):
    def __init__(self):
        super(Img2CDDD, self).__init__()
        # Input size: [batch, 3, 500, 500]

        self.network = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size = 7, stride=3, padding = 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size = 5, stride=1, padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, kernel_size = 5, stride=1, padding = 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(384, 384, kernel_size = 3, stride=1, padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size = 3, stride=1, padding = 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(384, 512, kernel_size = 3, stride=1, padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size = 3, stride=1, padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size = 3, stride=1, padding = 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Flatten(),

            nn.Linear(512*9*9, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, 512),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.network(x)

# Create the model

model = Img2CDDD()
model.eval()
model.to(device)

learning_rate = 1e-4
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))

gc.collect()

st.title("SMAI - Img2Mol")

file_list = st.file_uploader("Upload an image of the chemical", type=["jpg", "png"], accept_multiple_files=True)

if st.button("Predict"):
    gc.collect()
    images = torch.cat([torch.unsqueeze(load_image(file), 0) for file in file_list], dim=0)
    # image = torch.unsqueeze(load_image(file), 0)
    torch.save(images, IMAGE_TENSOR_FILE)
    
    # Test the model

    testloader = torch.utils.data.DataLoader(images)
    test_iter = iter(testloader)
    test_images = next(test_iter)
    test_images = test_images.to(device)

    for i, batch_x in enumerate(testloader, 0):
        batch_x = batch_x.to(device)
        predictions = model(batch_x)
        pred_np = predictions.cpu().detach().numpy()
        # data = {'cddd': pred_np.to_list()}
        # r = requests.post("http://localhost:8000/predict", json=data)
        # print(r.content)
        # embs = pred_np.tolist()
        # # infer_model = InferenceModel(model_dir="/home/reshyurem/cddd/default_model/")
        for emb in pred_np:
            # req = json.dumps({"cddd": pred_np.tolist()})
            # data = {'cddd': emb.tolist(), 'seq':"Hi"}
            r = requests.post("http://localhost:8000/predict", data=json.dumps({'cddd': emb.tolist(), 'seq': 'Konichiwa MFs'}))
            st.write(json.loads(r.text)['seq'])
            # print(json.loads(r.content.decode("utf-8")))
