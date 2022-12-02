import gc
import os
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from matplotlib import pyplot as plt
from PIL import Image, ImageOps, ImageEnhance
from pytorch_lightning import LightningModule as LM

# Options

opt = {
    'image_size': 224,
    'is_grayscale': False,
    'image_parsed': False,
    'dataset_size': 5000
}

# PATHS

ABSOLUTE_PATH = '/home/reshyurem/Accurate-smiles-recognition-Img2Mol/'
MODEL_FILE = ABSOLUTE_PATH + 'data/50000_images_30_epochs.pt'
IMG2MOL_IMAGE_PATH = ABSOLUTE_PATH + 'data/Img2Mol'
IMG2MOL_CDDD_PATH = ABSOLUTE_PATH + 'data/Img2Mol_embeddings.csv'

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

# Load the model

model.load_state_dict(torch.load(MODEL_FILE))

# Load the images and cddd values

indices = range(5000)
image_list = []
for i in indices:
    image_list.append(os.path.join(IMG2MOL_IMAGE_PATH,"{}.png".format(i)))
img2mol_images = torch.cat([torch.unsqueeze(load_image(image), 0) for image in image_list], dim=0)

img2mol_cddd_df = pd.read_csv(IMG2MOL_CDDD_PATH, delimiter=',',  header=None)
img2mol_cddd = torch.from_numpy(img2mol_cddd_df.values).float().to(device)
img2mol_cddd = img2mol_cddd[:opt['dataset_size']]

b_dataset = []
for i in range(len(img2mol_images)):
    b_dataset.append([img2mol_images[i], img2mol_cddd[i]])

benchmarkloader = torch.utils.data.DataLoader(b_dataset, batch_size=128,shuffle=True)

# Test the model on the values

for i, (batch_x, test_labels) in enumerate(benchmarkloader, 0):
    batch_x = batch_x.to(device)
    test_labels = test_labels.to(device)
    predictions = model(batch_x)
    loss = criterion(predictions[0], test_labels[0])
    print(i, loss)
    pred_np = predictions.cpu().detach().numpy()
    test_np = test_labels.cpu().detach().numpy()
    file1 = open("img2mol_benchmark_predictions.pkl", "wb")
    np.save(file1, pred_np, allow_pickle=False)
    file1.close()
    file2 = open("img2mol_benchmark_test_labels.pkl", "wb")
    np.save(file2, test_np, allow_pickle=False)
    file2.close()
    break