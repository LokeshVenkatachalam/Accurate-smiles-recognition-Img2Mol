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
    'dataset_size': 50000,
    'train': True,
    'epochs': 30,
    'load': False
}

# PATHS

ABSOLUTE_PATH = '/home/reshyurem/Accurate-smiles-recognition-Img2Mol/'
IMAGE_PATH = ABSOLUTE_PATH + 'data/train_images/'
IMAGE_TENSOR_FILE = IMAGE_PATH + 'image_tensor.pt'
CDDD_PATH = ABSOLUTE_PATH + 'data/embeddings.csv'
MODEL_FILE = ABSOLUTE_PATH + 'data/50000_images_30_epochs.pt'

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

# Load CDDD embeddings for training and testing

cddd_df = pd.read_csv(CDDD_PATH, delimiter=',',  header=None)
cddd = torch.from_numpy(cddd_df.values).float().to(device)
cddd = cddd[:opt['dataset_size']]

# Load corresponding images for training and testing
# Load the images from a tensor file if it exists

images = []

if opt['image_parsed']:
    images = torch.load(IMAGE_TENSOR_FILE)
else:
    indices = range(cddd_df.shape[0])
    image_list = []
    for i in indices:
        image_list.append(os.path.join(IMAGE_PATH,"{}.png".format(i)))
    images = torch.cat([torch.unsqueeze(load_image(image), 0) for image in image_list[:opt['dataset_size']]], dim=0)
    torch.save(images, IMAGE_TENSOR_FILE)

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

# Load the model if needed

if opt["load"] == True:
    model.load_state_dict(torch.load(MODEL_FILE))

# Split the dataset into training and testing and validation

dataset = []
for i in range(len(images)):
    dataset.append([images[i], cddd[i]])
train_size = int(0.8 * len(images))
test_size = len(images) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(train_size*0.9), int(train_size*0.1)])

# Train the model

gc.collect()
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128,shuffle=True)

if opt["train"] == True:
    for epoch in tqdm(range(opt['epochs'])):
        running_loss = 0.0
        st = time.time()
        for i, (batch_x, batch_y) in enumerate(trainloader, 0):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.data
            if i % 50 == 49:
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0
            print("Done with Batch {}, epoch {}".format(i, epoch))
        et = time.time()
        torch.save(model.state_dict(), MODEL_FILE)
        print(et-st, "Epoch", epoch)

    print('Finished Training')
    print('Saving Model...')
    torch.save(model.state_dict(), MODEL_FILE)

# Test the model

testloader = torch.utils.data.DataLoader(test_dataset, batch_size=128,shuffle=True)
test_iter = iter(testloader)
test_images, test_labels = next(test_iter)
test_images = test_images.to(device)

# Calculate MSE loss

for i, (batch_x, test_labels) in enumerate(testloader, 0):
    batch_x = batch_x.to(device)
    test_labels = test_labels.to(device)
    predictions = model(batch_x)
    loss = criterion(predictions[0], test_labels[0])
    print(i, loss)
    pred_np = predictions.cpu().detach().numpy()
    test_np = test_labels.cpu().detach().numpy()
    file1 = open(ABSOLUTE_PATH + "data/test_results/predictions_"+str(i)+".pkl", "wb")
    np.save(file1, pred_np, allow_pickle=False)
    file1.close()
    file2 = open(ABSOLUTE_PATH + "data/test_results/test_labels_"+str(i)+".pkl", "wb")
    np.save(file2, test_np, allow_pickle=False)
    file2.close()