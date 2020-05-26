import pickle
import operator

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import chain
from os.path import isfile
from PIL import Image as pil_image
from tqdm import tqdm_notebook, tqdm
from sklearn.metrics.pairwise import cosine_similarity

from torch.utils import data
from torch.optim.lr_scheduler import StepLR
from torchvision import models
from torchvision import transforms

### TENSORBOARD
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


from datetime import datetime
#datetime.now()

MODEL = "res50"
EPOCHS = 15 
EMBEDDINGS_SIZE = 1000
# Writer will output to ./runs/ directory by default

sum_writer = "logs/"+MODEL+"--ep_"+str(EPOCHS)+"--D_"+str(EMBEDDINGS_SIZE)+"--TIME_"+str(datetime.now())
comments = "epochs-"+str(EPOCHS)+"__contrastive-loss"


print("Summary Writer--  ", sum_writer, "   additional-comments-- ", comments)
writer = SummaryWriter(sum_writer, comment=comments)



# Avoid pil error
pil_image.MAX_IMAGE_PIXELS = None

# For reproducibility purpose
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_df_path = 'train.csv'
test_df_path = 'sample_submission.csv'
TRAIN = 'train/train/'
TEST = 'test/test/'

rotate_path = 'resnet-pytorch-files/rotate.txt'
exclude_path = 'resnet-pytorch-files/exclude.txt'
bboxes_path = 'resnet-pytorch-files/bounding-box.pickle'




train_df = pd.read_csv(train_df_path, header=0)
test_df = pd.read_csv(test_df_path)



NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]


def expand_path(p):
    '''Takes image name and returns full path'''
    if isfile(TRAIN + p):
        return TRAIN + p
    if isfile(TEST + p):
        return TEST + p
    return p

'''
def show_whale(imgs, per_row=2):
    n = len(imgs)
    rows = (n + per_row - 1) // per_row
    cols = min(per_row, n)
    fig, axes = plt.subplots(rows, cols, figsize=(24 // per_row * cols, 24 // per_row * rows))
    
    for ax in axes.flatten(): 
        ax.axis('off')
    for i, (img, ax) in enumerate(zip(imgs, axes.flatten())): 
        ax.imshow(img.convert('RGB'))
'''

# Load array of images to rotate
# https://www.kaggle.com/martinpiotte/humpback-whale-identification-model-files
with open(rotate_path, 'rt') as f: 
    rotate = set(f.read().split('\n')[:-1])
    
# Load array of images to exclude
# https://www.kaggle.com/martinpiotte/humpback-whale-identification-model-files
with open(exclude_path, 'rt') as f: 
    exclude = f.read().split('\n')[:-1]   
    
# Load bounding boxes data
# https://www.kaggle.com/martinpiotte/humpback-whale-identification-model-files
with open(bboxes_path, 'rb') as f:
    bboxes = pickle.load(f)

# Excluded images
train_df = train_df[~train_df['Image'].isin(exclude)]

train_df = train_df[train_df['Id'] != 'new_whale'].reset_index(drop=True)


## IMAGE LOADING

# The margin added around the bounding box to compensate for bounding box inaccuracy
IMAGE_SIZE = (224, 224)
crop_margin  = 0.05 
resize = transforms.Resize(IMAGE_SIZE)

def load_image(image_name):
    image = pil_image.open(expand_path(image_name)).convert('RGB')
    width, height = image.size

    # Crop bounding box with respect to crop margin
    x0, y0, x1, y1 = bboxes[image_name]
    dx = x1 - x0
    dy = y1 - y0
    x0 -= dx * crop_margin
    x1 += dx * crop_margin + 1
    y0 -= dy * crop_margin
    y1 += dy * crop_margin + 1
    if (x0 < 0):
        x0 = 0
    if (x1 > width):
        x1 = width
    if (y0 < 0):
        y0 = 0
    if (y1 > height):
        y1 = height
    try:
        # A few images have incorrect bounding boxes
        image = image.crop((x0, y0, x1, y1))
    except:
        pass
    # Rotate whale tails which are upside-down
    if image_name in rotate:
        image = image.rotate(180)
        
    return resize(image)


## Cache training images to speedup train dataloader

cache = {}

for image_name in tqdm_notebook(train_df['Image']):
    cache[image_name] = load_image(image_name)
    
def get_image(image_name):
    '''Returns cropped and resized image either from cache or from disk'''
    return cache.get(image_name) or load_image(image_name)


### Training/Validating Datasets

class WhaleDataset(data.Dataset):
    '''
    PyTorch class for Whales' tails
    Link: https://www.kaggle.com/c/whale-categorization-playground/data

    Data issues mentioned:
    - some bboxes are incorrect
    '''

    def __init__(self, images_data, scope='train', augment=False):
        self.image_data = images_data
        self.scope = scope

        if augment:
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0, contrast=0.05, saturation=0.05),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD)
            ])

        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD)
            ])

        print('Images: {}. Augmentation: {}. Scope: {}.'.format(len(self.image_data), augment, scope))

    def __getitem__(self, idx):

        '''
        For train and validation triplets are required, for prediction - only images;
        '''
        image_mode = 'RGB'
        row = self.image_data.iloc[idx]
        anchor_name = row['Image']
        anchor = get_image(anchor_name)

        anchor = self.transform(anchor)
        if self.scope == 'train' or self.scope == 'val':
            anchor_id = row['Id']

            positive_candidates = list(self.image_data[self.image_data['Id'] == anchor_id]['Image'])
            positive_candidates = [x for x in positive_candidates if x != anchor_name]

            if len(positive_candidates) == 0:
                positive_name = anchor_name
            else:
                positive_name = np.random.choice(positive_candidates)

            negative_candidates = list(
                self.image_data[(self.image_data['Id'] != anchor_id)]['Image']
            )
            negative_name = np.random.choice(negative_candidates)

            positive = get_image(positive_name)
            negative = get_image(negative_name)

            positive = self.transform(positive)
            negative = self.transform(negative)

            return {'name': anchor_name,
                    'anchor': anchor,
                    'positive': positive,
                    'negative': negative
                    }
        else:
            return {'name': anchor_name, 'anchor': anchor}
    

    def __len__(self):
        return len(self.image_data)


### Siamese NN validation is a little bit tricky.
### We will take only images from classes with more than 3 images per class.
 

# Group images by id
grouped = train_df.groupby('Id')

validation_indexes = []

for group in grouped.groups.items():
    indexes = group[1]
    # Take only one image from class which has at least 3 images
    if (len(indexes) > 2):
        validation_indexes.append(indexes[0])
        
validation_df = train_df.iloc[validation_indexes].reset_index(drop=True)
train_df = train_df[~train_df.index.isin(validation_indexes)].reset_index(drop=True)


train_dataset = WhaleDataset(train_df, augment=True)
train_dataloader = data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers = 4, drop_last=False)

validation_df = WhaleDataset(validation_df)
validation_dataloader = data.DataLoader(validation_df, batch_size=32, shuffle=False, num_workers = 4, drop_last=False)


### Pretrained Resnet18/34/50  

class ResNet34(nn.Module):
    '''
    Last fully connected layer changed to ouput EMBEDDINGS_SIZE-dim vector.
    '''
    
    def __init__(self):
        super(ResNet34, self).__init__()
        self.model = models.resnet34(pretrained=True)
        self.model.fc = nn.Linear(in_features=512, out_features=EMBEDDINGS_SIZE, bias=True)

    def forward(self, image):
        features = self.model(image)
        return features

class ResNet18(nn.Module):
    '''
    Last fully connected layer changed to ouput EMBEDDINGS_SIZE-dim vector.
    '''

    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(in_features=512, out_features=EMBEDDINGS_SIZE, bias=True)

    def forward(self, image):
        features = self.model(image)
        return features

class ResNet50(nn.Module):
    '''
    Last fully connected layer changed to ouput EMBEDDINGS_SIZE-dim vector.
    '''

    def __init__(self):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=True)
        #self.model.fc = nn.Linear(in_features=2048, out_features=1000, bias=True)
        self.model.fc = nn.Linear(in_features=2048, out_features=EMBEDDINGS_SIZE, bias=True)

    def forward(self, image):
        features = self.model(image)
        return features

if MODEL == "res50":
 model = ResNet50().cuda()
elif MODEL == "res34":
 model = ResNet34().cuda()
elif MODEL == "res18":
 model = ResNet18().cuda()
### Define Triplet loss with margin 0.7

class TripletLossCosine(nn.Module):
    def __init__(self):
        super(TripletLossCosine, self).__init__()
        self.MARGIN = 0.7
            
    def forward(self, anchor, positive, negative):
        dist_to_positive = 1 - F.cosine_similarity(anchor, positive)
        dist_to_negative = 1 - F.cosine_similarity(anchor, negative)
        loss = F.relu(dist_to_positive - dist_to_negative + self.MARGIN)
        loss = loss.mean()
        return loss
    
loss_func = TripletLossCosine()

## Optimizer!!!
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.007, momentum=0.9)
lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.4)

### Training

def validate(model): 
    model.eval()
    batch_losses = []

    with torch.no_grad():
        for sample in validation_dataloader:
            for key in ['anchor','positive','negative']:
                sample[key] = sample[key].cuda()

        anchor_embed = model(sample['anchor'])
        positive_embed = model(sample['positive'])
        negative_embed = model(sample['negative'])
        loss = loss_func(anchor_embed, positive_embed, negative_embed) 

        batch_losses.append(loss.item())
    return batch_losses


#### TRAINING

train_losses = []
validation_losses = []

for epoch in range(1, EPOCHS+1):
    batch_losses = []

    for sample in tqdm_notebook(train_dataloader):
        model.train()
        optimizer.zero_grad()

        for key in ['anchor','positive','negative']:
            sample[key] = sample[key].cuda()

        anchor_embed = model(sample['anchor'])
        positive_embed = model(sample['positive'])
        negative_embed = model(sample['negative'])
        loss = loss_func(anchor_embed, positive_embed, negative_embed)  
        loss.backward()
        
        optimizer.step()
        if epoch == 10:
            optimizer = torch.optim.SGD(params=model.parameters(), lr=0.0005, momentum=0.9)

#         lr_scheduler.step()
        batch_losses.append(loss.item())

    mean_loss = np.mean(batch_losses)
    #print("data: ", type(mean_loss), " value: ", mean_loss)
    
    writer.add_scalar('Loss/train', mean_loss, epoch)        
    train_losses.append(mean_loss)
	    
    val_loss = validate(model)
    val_loss_mean = np.mean(val_loss)
    validation_losses.append(val_loss_mean)
    writer.add_scalar('Loss/validation', val_loss_mean, epoch)
    print('====Epoch {}. Train loss: {}. Val loss: {}'.format(epoch,  mean_loss,  val_loss_mean))


all_images = pd.DataFrame({'Image': pd.concat([train_df['Image'], test_df['Image']])}).reset_index(drop=True)

embed_dataset = WhaleDataset(all_images, scope='embed')
embed_dataloader = data.DataLoader(embed_dataset, batch_size=32, shuffle=False, num_workers = 4, drop_last=False)
embed_dataset = WhaleDataset(all_images, scope='embed')
print(type(embed_dataset))
embed_dataloader = data.DataLoader(embed_dataset, batch_size=32, shuffle=False, num_workers = 4, drop_last=False)
embeddings_dict = {}

#final_embeddings = np.matrix((len(all_images), EMBEDDINGS_SIZE))
#final_anchors = np.matrix((len(all_images), 3, 224, 224))
#final_dict = np.matrix(len(all_images)) 

##
c, h, w =  list(embed_dataset[0]['anchor'].size())
show_em = torch.ones((1000, EMBEDDINGS_SIZE), dtype=torch.float32)
show_anchors = torch.ones ( ( 1000, c, h, w ), dtype=torch.float32 )


model.eval()
with torch.no_grad():
    for sample in tqdm_notebook(embed_dataloader):
        anchors = sample['anchor'].cuda()
        embeds = model(anchors)

        #print("embdeds "+ str(embeds.size()))
        #final_embeddings.append(embeds, axis=0)
        #np.append(final_anchors, anchors, axis=0)
        #final_dict.append(
        writer.add_embedding(embeds, label_img = anchors)
 
        for image_name, embed in zip(sample['name'], embeds):
            embeddings_dict[image_name] = embed.cpu().numpy()
            

# Due to memory error we will show case 1000 embeddings
for i in range(0,1000):
 show_em[i] = torch.from_numpy(embeddings_dict[train_df['Image'][i]])
 show_anchors[i] = embed_dataset[i]['anchor']

assert len(embeddings_dict) == len(all_images)

writer.add_embedding(show_em, label_img=show_anchors)

del show_em
del show_anchors

train_embeds = np.zeros((len(train_df), EMBEDDINGS_SIZE))
test_embeds = np.zeros((len(test_df), EMBEDDINGS_SIZE))

for index, image_name in enumerate(train_df['Image']):
    train_embeds[index] = embeddings_dict[image_name]
#print(type(embeddings_dict))

for index, image_name in enumerate(test_df['Image']):
    test_embeds[index] = embeddings_dict[image_name]
#print(type(test_embeds[0]))

print(model)
#writer.add_graph(model)
writer.close()
#print("final-anchors: "+ np.shape(final_anchors))
#writer.add_embedding(train_embeds, label_img=final_anchors, metadata = train_df['Image'])


similarities = cosine_similarity(test_embeds, train_embeds)


def find_top_k(sims, k):
    top_sims = sims.argsort()[::-1]
    top_klasses = set(['new_whale'])
    for sim in top_sims:
        klass = train_df.iloc[sim]['Id']
        top_klasses.add(klass)
        if len(top_klasses) == k:
            break
    return ' '.join(top_klasses)


find_top_k(similarities[666], 5)


for index in tqdm_notebook(range(len(test_df))):
    test_df.iloc[index]['Id'] = find_top_k(similarities[index], 5)
#    print(test_df.iloc[index]['Image'])

test_df.to_csv('submission.csv', index=False)

