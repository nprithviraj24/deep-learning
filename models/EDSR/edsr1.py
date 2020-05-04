import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models.vgg as vgg
# visualizing data
import matplotlib.pyplot as plt
import numpy as np
import warnings
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("Cycle-EDSR/test")



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import math

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

def get_data_loader(image_type, image_dir='lrtohr', image_size=64, batch_size=8, num_workers=0):
    """Returns training and test data loaders for a given image type
    """

    # resize and normalize the images
    transform1 = transforms.Compose([transforms.Resize((image_size, image_size)), # resize to 128x128
                                    transforms.ToTensor()])
    # get training and test directories
    # resize and normalize the images
    transform2 = transforms.Compose([transforms.Resize((256,256)), # resize to 128x128
                                    transforms.ToTensor()])

    image_path = './' + image_dir
    train_path = os.path.join(image_path, image_type)
    test_path = os.path.join(image_path, 'test_{}'.format(image_type))

    if image_type == 'lr':
        # define datasets using ImageFolder
        train_dataset = datasets.ImageFolder(train_path, transform1)
        test_dataset = datasets.ImageFolder(test_path, transform1)

        # create and return DataLoaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if image_type == 'hr':
        # define datasets using ImageFolder
        train_dataset = datasets.ImageFolder(train_path, transform2)
        test_dataset = datasets.ImageFolder(test_path, transform2)

            # create and return DataLoaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


# Create train and test dataloaders for images from the two domains X and Y
# image_type = directory names for our data
# del dataloader_X, test_dataloader_X
# del dataloader_Y, test_dataloader_Y

dataloader_X, test_iter_X = get_data_loader(image_type='lr')
dataloader_Y, test_iter_Y = get_data_loader(image_type='hr')

print(convert_size(torch.cuda.memory_allocated(device=device)))



# helper scale function
def scale(x, feature_range=(0, 1)):
    ''' Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1.
       This function assumes that the input x is already scaled from 0-1.'''

    # scale from 0-1 to feature_range
    # min, max = feature_range
    # x = x * (max - min) + min
    return x



import torch.nn as nn
import torch.nn.functional as F

# helper conv function
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class DOne(nn.Module):

    def __init__(self, conv_dim=64):
        super(DOne, self).__init__()

        # Define all convolutional layers
        # Should accept an RGB image as input and output a single value

        # Convolutional layers, increasing in depth
        # first layer has *no* batchnorm
        self.conv1 = conv(3, conv_dim, 4, batch_norm=False) # x, y = 256, depth 256
        self.conv2 = conv(conv_dim, conv_dim*2, 4) # (128, 128, 128)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4) # (64, 64, 256)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4) # (32, 32, 512)
        self.conv5 = conv(conv_dim*8, conv_dim*16, 4) # (16, 16, 1024)
        self.conv6 = conv(conv_dim*16, conv_dim*32, 4) # (8, 8, 2048)
        self.conv7 = conv(conv_dim*32, conv_dim*64, 4) # (4, 4, 4096)

        # Classification layer
        self.conv8 = conv(conv_dim*64, 1, 4, stride=1, batch_norm=False)

    def forward(self, x):
        # relu applied to all conv layers but last
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        out = F.relu(self.conv7(out))
        # last, classification layer
        out = self.conv8(out)
        # print(type(self.conv1))
        return out

class DTwo(nn.Module):

    def __init__(self, conv_dim=64):
        super(DTwo, self).__init__()

        # Define all convolutional layers
        # Should accept an RGB image as input and output a single value

        # Convolutional layers, increasing in depth
        # first layer has *no* batchnorm
        self.conv1 = conv(3, conv_dim, 4, batch_norm=False) # x, y = 64, depth 64
        self.conv2 = conv(conv_dim, conv_dim*2, 4) # (32, 32, 128)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4) # (16, 16, 256)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4) # (8, 8, 512)
        self.conv5 = conv(conv_dim*8, conv_dim*16, 4) # (4, 4, 1024)
        # self.conv6 = conv(conv_dim*16, conv_dim*32, 4) # (8, 8, 2048)
        # self.conv7 = conv(conv_dim*32, conv_dim*64, 4) # (4, 4, 4096)

        # Classification layer
        self.conv8 = conv(conv_dim*16, 1, 4, stride=1, batch_norm=False)

    def forward(self, x):
        # relu applied to all conv layers but last
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        # out = F.relu(self.conv6(out))
        # out = F.relu(self.conv7(out))
        # last, classification layer
        out = self.conv8(out)
        # print(type(self.conv1))
        return out


# residual block class
class ResidualBlock(nn.Module):
    """Defines a residual block.
       This adds an input x to a convolutional layer (applied to x) with the same size input and output.
       These blocks allow a model to learn an effective transformation from one domain to another.
    """
    def __init__(self, conv_dim, bn=True):
        super(ResidualBlock, self).__init__()
        # conv_dim = number of inputs
        
        # define two convolutional layers + batch normalization that will act as our residual function, F(x)
        # layers should have the same shape input as output; I suggest a kernel_size of 3
        
        self.conv_layer1 = conv(in_channels=conv_dim, out_channels=conv_dim, 
                                kernel_size=3, stride=1, padding=1, batch_norm=bn)
        
        self.conv_layer2 = conv(in_channels=conv_dim, out_channels=conv_dim, 
                               kernel_size=3, stride=1, padding=1, batch_norm=bn)
        
    def forward(self, x):
        # apply a ReLu activation the outputs of the first layer
        # return a summed output, x + resnet_block(x)
        out_1 = F.relu(self.conv_layer1(x))
        out_2 = x + self.conv_layer2(out_1)
        return out_2



from collections import namedtuple
LossOutput = namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
# https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
    
    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return LossOutput(**output)



c = 128  #initially 256 

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, sign):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = float(sign) * torch.Tensor(rgb_mean)

        # Freeze the MeanShift layer
        for params in self.parameters():
            params.requires_grad = False

class _Residual_Block(nn.Module): 
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x): 
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output *= 0.1
        output = torch.add(output,identity_data)
        return output

 
class GOne(nn.Module):
    def __init__(self):
        super(GOne, self).__init__()

        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.sub_mean = MeanShift(rgb_mean, -1)

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=c, kernel_size=3, stride=1, padding=1, bias=False)

        self.residual = self.make_layer(_Residual_Block, 32)

        self.conv_mid = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1, bias=False)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels=c, out_channels=c*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )
        self.upsample2x = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv_output = nn.Conv2d(in_channels=c, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)

        self.add_mean = MeanShift(rgb_mean, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        # out = self.sub_mean(x)
        out = self.conv_input(x)
        residual = out
        # print("inside")
        out = self.conv_mid(self.residual(out))
        out = torch.add(out,residual)
        # out = self.upscale4x(out)
        out = self.upsample2x(out)
        out = self.upsample2x(out)
        out = self.conv_output(out)
        # out = self.add_mean(out)
        return out


c = 64  #initially 256 
batchnorm = True
kernels = [3,3] 


class GTwo(nn.Module):
       def __init__(self):
         super(GTwo, self).__init__()
         self.conv1 = nn.Conv2d(3, c, kernel_size=kernels[0], stride=1, padding=1)

         self.conv2 = nn.Conv2d(c, c, kernel_size=kernels[1], stride=1, padding=1)

         self.conv3 = nn.Conv2d(c, c, kernel_size=kernels[1], stride=2, padding=1)

         self.residual = self.make_res_layers(ResidualBlock, 6, 64)

         self.conv4 = nn.Conv2d(c, c, kernel_size=kernels[1], stride=2, padding=1)

         self.conv5 = nn.Conv2d(c, c, kernel_size=kernels[1], stride=1, padding=1)

         self.conv6 = nn.Conv2d(c, 3, kernel_size=kernels[0], stride=1, padding=1)
       

       def make_res_layers(self, block, num_of_layer, conv_dim):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(conv_dim, bn=batchnorm))
        return nn.Sequential(*layers)

       def forward(self, x):
          out = F.relu(self.conv1(x))
          out = F.relu(self.conv2(out))
          out = F.relu(self.conv3(out))
          residual = self.residual(out)
          out = torch.add(out,residual)
          # out = self.upscale4x(out)
          out = F.relu(self.conv4(out))
          out = F.relu(self.conv5(out))
          out = F.relu(self.conv6(out))
          return out



def create_model(g_conv_dim=64, d_conv_dim=64, n_res_blocks=6):
    """Builds the generators and discriminators."""
    
    # Instantiate generators
    G_XtoY = GOne()
    G_YtoX = GTwo()
    # Instantiate discriminators
    D_X = DTwo(64)
    D_Y = DOne(64)

    # move models to GPU, if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        G_XtoY.to(device)
        G_YtoX.to(device)
        D_X.to(device)
        D_Y.to(device)
        print('Models moved to GPU.')
    else:
        print('Only CPU available.')

    return G_XtoY, G_YtoX, D_X, D_Y

# call the function to get models
G_XtoY, G_YtoX, D_X, D_Y = create_model()

# print all of the models
#print_models(G_XtoY, G_YtoX, D_X, D_Y)

## LOSS FUnctions
def tv_loss(batch, weight):
    return weight * (
    torch.sum(torch.abs(batch[:, :, :, :-1] - batch[:, :, :, 1:])) + 
    torch.sum(torch.abs(batch[:, :, :-1, :] - batch[:, :, 1:, :]))
)  

def real_mse_loss(D_out):
    # how close is the produced output from being "real"?
    return torch.mean((D_out-1)**2)

def fake_mse_loss(D_out):
    # how close is the produced output from being "fake"?
    return torch.mean(D_out**2)

def cycle_consistency_loss(real_im, reconstructed_im, lambda_weight):
    # calculate reconstruction loss
    # as absolute value difference between the real and reconstructed images
    reconstr_loss = torch.mean(torch.abs(real_im - reconstructed_im))
    # return weighted loss
    return lambda_weight*reconstr_loss

mse_loss = torch.nn.MSELoss()



import torch.optim as optim

# hyperparams for Adam optimizer
lr=0.0002
beta1=0.5
beta2=0.999 # default value

g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())  # Get generator parameters

# Create optimizers for the generators and discriminators
g1_optimizer = optim.Adam(G_XtoY.parameters(), lr, [beta1, beta2])
g_optimizer = optim.Adam(g_params, lr, [beta1, beta2])
g2_optimizer = optim.Adam(G_YtoX.parameters(), lr, [beta1, beta2])
d_x_optimizer = optim.Adam(D_X.parameters(), lr, [beta1, beta2])
d_y_optimizer = optim.Adam(D_Y.parameters(), lr, [beta1, beta2])

vgg_model = vgg.vgg16()
state = torch.load('checkpoints/vgg16_397923af.pth') 
vgg_model.load_state_dict(state)
if torch.cuda.is_available():
  vgg_model.to(device)
loss_network = LossNetwork(vgg_model)
loss_network.eval()


def training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y,
                  n_epochs=1000):

    print_every=10

    # keep track of losses over time
    losses = []

    test_iter_X = iter(test_dataloader_X)
    test_iter_Y = iter(test_dataloader_Y)
    # print("0")
    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    # make sure to scale to a range -1 to 1
    fixed_X = scale(test_iter_X.next()[0])
    fixed_Y = scale(test_iter_Y.next()[0])
    # fixed_X = scale(fixed_X) 
    # fixed_Y = scale(fixed_Y)

    # batches per epoch
    iter_X = iter(dataloader_X)
    iter_Y = iter(dataloader_Y)
    batches_per_epoch = min(len(iter_X), len(iter_Y))

    for epoch in range(1, n_epochs+1):

        # Reset iterators for each epoch
        if epoch % batches_per_epoch == 0:
            iter_X = iter(dataloader_X)
            iter_Y = iter(dataloader_Y)

        images_X, _ = iter_X.next()
        images_X = scale(images_X) # make sure to scale to a range -1 to 1

        images_Y, _ = iter_Y.next()
        images_Y = scale(images_Y)

        # move images to GPU if available (otherwise stay on CPU)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        images_X = images_X.to(device)
        images_Y = images_Y.to(device)
        # print(images_X.shape, "    ",  images_Y.shape )
        # print(epoch, " -- ", images_X.shape)


        # ============================================
        #            TRAIN THE DISCRIMINATORS
        # ============================================

        ##   First: D_X, real and fake loss components   ##

        # Train with real images
        d_x_optimizer.zero_grad()

        # 1. Compute the discriminator losses on real images
        out_x = D_X(images_X)
        # print("1.")
        D_X_real_loss = real_mse_loss(out_x)
        # print("1: ", convert_size(torch.cuda.memory_allocated(device=device)), end="  ")
        
        # Train with fake images

        # 2. Generate fake images that look like domain X based on real images in domain Y
        fake_X = G_YtoX(images_Y)
        # print("2: ", convert_size(torch.cuda.memory_allocated(device=device)), end="  ")
        # # print("2.")
        # 3. Compute the fake loss for D_X
        out_x = D_X(fake_X)
        D_X_fake_loss = fake_mse_loss(out_x)
        # print("3: ", convert_size(torch.cuda.memory_allocated(device=device)), end="  ")
        # # del out_x

        # 4. Compute the total loss and perform backprop
        d_x_loss = D_X_real_loss + D_X_fake_loss
        d_x_loss.backward()
        d_x_optimizer.step()


        ##   Second: D_Y, real and fake loss components   ##

        # Train with real images
        d_y_optimizer.zero_grad()

        # 1. Compute the discriminator losses on real images
        out_y = D_Y(images_Y)
        D_Y_real_loss = real_mse_loss(out_y)
        # print("4: ", convert_size(torch.cuda.memory_allocated(device=device)), end="  ")
        # del out_y
        # Train with fake images

        # 2. Generate fake images that look like domain Y based on real images in domain X
        fake_Y = G_XtoY(images_X)
        # print("5: ", convert_size(torch.cuda.memory_allocated(device=device)), end="  ")
        # 3. Compute the fake loss for D_Y
        out_y = D_Y(fake_Y)
        D_Y_fake_loss = fake_mse_loss(out_y)

        # print("6: ", convert_size(torch.cuda.memory_allocated(device=device)), end="  ")
        # del out_y
        # 4. Compute the total loss and perform backprop
        d_y_loss = D_Y_real_loss + D_Y_fake_loss
        d_y_loss.backward()
        d_y_optimizer.step()


        # =========================================
        #            TRAIN THE GENERATORS
        # =========================================

        ##    First: generate fake X images and reconstructed Y images    ##
        # g1_optimizer.zero_grad()
        # g2_optimizer.zero_grad()
        g_optimizer.zero_grad()


        # 1. Generate fake images that look like domain X based on real images in domain Y
        fake_X = G_YtoX(images_Y)
        # print("6.")
        # 2. Compute the generator loss based on domain X
        out_x = D_X(fake_X)
        # print("7.")
        g_YtoX_loss = real_mse_loss(out_x)
        del out_x
        # writer.add_scalar("MSE/Y-to-X", g_YtoX_loss, epoch)
        # 3. Create a reconstructed y
        # 4. Compute the cycle consistency loss (the reconstruction loss)
        reconstructed_Y = G_XtoY(fake_X)
        # print("8.")
        reconstructed_y_loss = cycle_consistency_loss(images_Y, reconstructed_Y, lambda_weight=50)
        

        featuresX = loss_network(images_X)
        featuresFakeX = loss_network(fake_X)
        CONTENT_WEIGHT = 0.5
        # print(" features:  batch--- ", featuresY[1].data.shape, "    ",  featuresFakeY[1].data.shape )

        content_loss_X = CONTENT_WEIGHT * mse_loss(featuresX[1].data, featuresFakeX[1].data)
        del featuresX, featuresFakeX

        # g2_total_loss = g_YtoX_loss + reconstructed_y_loss




        ##    Second: generate fake Y images and reconstructed X images    ##

        # 1. Generate fake images that look like domain Y based on real images in domain X
        fake_Y = G_XtoY(images_X)
        # print("7: ", convert_size(torch.cuda.memory_allocated(device=device)), end="  ")
        # 2. Compute the generator loss based on domain Y
        out_y = D_Y(fake_Y)
        #print("8: ", convert_size(torch.cuda.memory_allocated(device=device)), end=" ")
        g_XtoY_loss = real_mse_loss(out_y)
        # writer.add_scalar("MSE/X-to-Y", g_XtoY_loss, epoch)
        del out_y
        # 3. Create a reconstructed x
        # 4. Compute the cycle consistency loss (the reconstruction loss)
        reconstructed_X = G_YtoX(fake_Y)
        #print("9: ", convert_size(torch.cuda.memory_allocated(device=device)))
        # print("11.")
        reconstructed_x_loss = cycle_consistency_loss(images_X, reconstructed_X, lambda_weight=50)

        # 5. Add up all generator and reconstructed losses and perform backprop
        

# xc = Variable(x.data, volatile=True)
        featuresY = loss_network(images_Y)
        featuresFakeY = loss_network(fake_Y)
        CONTENT_WEIGHT = 0.5
        # print(" features:  batch--- ", featuresY[1].data.shape, "    ",  featuresFakeY[1].data.shape )

        content_loss_Y = CONTENT_WEIGHT * mse_loss(featuresY[1].data, featuresFakeY[1].data)
        del featuresY, featuresFakeY
        downsample = nn.Upsample(scale_factor=0.25, mode='bicubic')
        identity_loss = mse_loss(downsample(fake_Y), images_X )
        g_total_loss = g_XtoY_loss + g_YtoX_loss + reconstructed_x_loss + reconstructed_y_loss + tv_loss(fake_Y, 0.5) + tv_loss(fake_X, 0.5) + content_loss_Y + content_loss_X + identity_loss

        g_total_loss.backward()
        g_optimizer.step()
        # g2_total_loss.backward()
        # g2_optimizer.step()
        


        # Print the log info
        if epoch % print_every == 0:
            # append real and fake discriminator losses and the generator loss
            losses.append((d_x_loss.item(), d_y_loss.item(), g_total_loss.item()))
            print('Epoch [{:5d}/{:5d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | g_total_loss: {:6.4f}'.format(epoch, n_epochs, d_x_loss.item(), d_y_loss.item(), g_total_loss.item()))
            # tb.save_value("Discriminator", "D_X", epoch, d_x_loss.item())
            # tb.save_value("Discriminator", "D_Y", epoch, d_y_loss.item())
            # tb.save_value("Generator", "", epoch, g1_total_loss.item())
#            dx = print(type(d_x_loss.item()))
            writer.add_scalar('Discriminator/D_X loss', d_x_loss, epoch)
            writer.add_scalar('Discriminator/D_Y loss', d_y_loss, epoch)
            writer.add_scalar('Generator', g_total_loss, epoch)

        sample_every=25
        # Save the generated samples
        if epoch % sample_every == 0:
            G_YtoX.eval() # set generators to eval mode for sample generation
            G_XtoY.eval()
            with torch.no_grad():
             fakeY = G_XtoY(fixed_X.to(device))
             fakeX = G_YtoX(fixed_Y.to(device))
            # fY = fake_Y[0].detach().cpu().resize_(1,256,256,3)
            # def to_data(x):
    # """Converts variable to numpy."""
             #fY = fakeY.cpu()
             #fY = fY.data.numpy()
             #fY = ((fY +1)*255 / (2)).astype(np.uint8) # rescale to 0-255
#            imshow(torchvision.utils.make_grid(torch.from_numpy(fY)))
             writer.add_image("Y/"+str(epoch), torchvision.utils.make_grid(fakeY.cpu()), global_step=epoch)
             #fX = fakeX.cpu()
             #fX = fX.data.numpy()
             #fX = ((fX +1)*255 / (2)).astype(np.uint8) # rescale to 0-255
             writer.add_image("X/"+str(epoch), torchvision.utils.make_grid(fakeX.cpu()), global_step=epoch)
 #           imshow(torchvision.utils.make_grid(torch.from_numpy(fX)))
             #del fY, fX
             del fakeY, fakeX
            # plt.torchvision.utils.make_grid(fake_Y.detach().cpu()))            # plt.imshow(fY)
            # save_samples(epoch, fixed_Y, fixed_X, G_YtoX, G_XtoY, batch_size=8)
            G_YtoX.train()
            G_XtoY.train()

        # uncomment these lines, if you want to save your model
        checkpoint_every = 1000
        # Save the model parameters
        if epoch % checkpoint_every == 0:
            torch.save({
            'G_XtoY_state_dict': G_XtoY.state_dict(),
            'G_YtoX_state_dict': G_YtoX.state_dict(),
            'D_X_state_dict': D_X.state_dict(),
            'D_Y_state_dict': D_Y.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'd_x_optimizer_state_dict': d_x_optimizer.state_dict(),
            'd_y_optimizer_state_dict': d_y_optimizer.state_dict()
            },"Cycle-EDSR/checkpoints/"+str(epoch)+".pth")
        torch.cuda.empty_cache()
    return losses


n_epochs = 10000 # keep this small when testing if a model first works

losses = training_loop(dataloader_X, dataloader_Y, test_iter_X, test_iter_Y, n_epochs=n_epochs)


