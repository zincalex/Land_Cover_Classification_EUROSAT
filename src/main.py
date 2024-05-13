# TODO chiedere al prof quale sia migliore/convenzione tra patterns, instances, samples, etc....
# fixare if che controlla de nel dataset ci sono anche altri file o solo directory
# dimensionality reduction, PCA

# All resnet50 pretrained models require 3 x 224 x 224 sizes
# Visto che il data set è diviso in cartelle, noi facciamo che prendiamo una cartella e prendiamo x% delle prime immagini
# Per ogni classe, creare le matrici di training con valore + label -----> assegnare noi le label
# fare shuffle una volta preso tutto il training set

# based 
import os               # paths
import numpy as np      # arrays 
import matplotlib.pyplot as plt 
import tifffile  
from tqdm import tqdm   # progress bar

# torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from torchvision.models import resnet50, ResNet50_Weights


DEVICE = (
    "cuda" 
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
torch.set_default_device(DEVICE)
if torch.cuda.is_available():
    g_device = torch.Generator(device='cuda')
if torch.backends.mps.is_available():
    g_device = torch.Generator(device='mps')
print(f"Using {DEVICE} device")




# CLASSES AND FUNCTIONS 
class EuroSATDataset(Dataset) : 
    """Eurosat dataset.

    Args:
        instances (list): list of instances
        labels (list): list of labels
        transform (callable): transform to apply to the instances
    """
    def __init__(self, instances, labels, transform):
        self.labels = labels                                     # class
        self.instances = instances                               # images
        self.transform = transform                               # transforms

    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        return self.transform(self.instances[idx]), self.labels[idx] 




def load_tif_channels(img_path, bands_selected):
    file = tifffile.imread(img_path)
    channels = file[..., bands_selected]
    channels = channels.astype(np.float32)/65535.0
    return channels



def main () : 

    # PARAMETERS
    input_size = (64, 64)                                               # image size
    bands = 13                                                          # number of channels
    fraction_train = 0.8                                                # training split
    fraction_test = 1 - fraction_train                                  # test split
    batch_size = 32                                                     # batch size
    lr = 1e-4                                                           # learning rate
    factor = 20                                                         # learning rate factor for tuning
    epochs = 2                                                          # fixed number of epochs
    #bands_selected = [0,1,2,3,4,5,6,7,8,9,10,11,12]                    # bands selected to analyze 4 3 2 is RGB 
    bands_selected = [4,3,2]


    # DATASET
    dataset_path = '../dataset/'
    if(os.path.isdir(dataset_path)):
        labels = os.listdir(dataset_path) 
    num_classes = len(labels)                   # number of classes, 10 for the EuroSAT dataset

    dict_class = {}
    for i in range(len(labels)) : 
        dict_class.update({labels[i] : i + 1})

    train_instances, train_label, test_instances, test_label = [], [], [], []
    with tqdm(total=num_classes,  unit='label') as pbar:
        for label in os.listdir(dataset_path) :
            label_dir = os.path.join(dataset_path, label)
            images = os.listdir(label_dir)

            m_training = int(len(images) * fraction_train)
            m_test = int(len(images) * fraction_test)
            for i in range(m_training) : 
                img_path = label_dir + '/' + images[i]
                #train_instances.append(Image.open(img_path).convert('RGB'))
                train_instances.append(load_tif_channels(img_path, bands_selected))
                train_label.append(dict_class[label])

            for i in range(m_training, len(images)) :               # the remaining images are for the test set 
                img_path = label_dir + '/' + images[i]
                #test_instances.append(Image.open(img_path).convert('RGB'))
                test_instances.append(load_tif_channels(img_path, bands_selected))
                test_label.append(dict_class[label])
            pbar.update(1)

        
    # Define transformations to apply to the images (e.g., resizing, normalization)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),                                                                 # Resize images to 256x256
        transforms.CenterCrop(224),                                                             # Crop the images to our desired size
        transforms.ToTensor(),                                                                  # Convert images to PyTorch tensors (standardization is automatically applied)
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])            # Normalize images
    ])
    
    train_dataset = EuroSATDataset(train_instances, train_label, transform)  
    test_dataset = EuroSATDataset(test_instances, test_label, transform)  
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, generator = g_device)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True, generator = g_device)
    

    # RESNET50 
    model = resnet50(weights = ResNet50_Weights.DEFAULT)
    model.to(DEVICE)
    loss_funct = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    with tqdm(total=epochs, unit='epoch') as pbar:
        for epoch in range(epochs):
            
            running_loss = 0.0
            with tqdm(total=len(train_data_loader), unit='instance') as inpbar:
                for i, data in enumerate(train_data_loader):
                    images, labels = data[0].to(DEVICE), data[1].to(DEVICE)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = loss_funct(outputs, labels)

                    # backward pass
                    loss.backward()
                    optimizer.step()
                    inpbar.update(1)
            pbar.update(1)
            print('Training loss: ', loss.item(), 'epoch: ', epoch)

    model.eval()

    total = 0
    correct_predictions = 0

    print("Starting testing")

    with tqdm(total=len(test_data_loader), unit='instance') as testbar:
        for i, test_data in enumerate(test_data_loader):
            images, labels = test_data[0].to(DEVICE), test_data[1].to(DEVICE)
            outputs = model(images)

            _,predicted = torch.max(outputs, 1) #max 1 probability, takes the max probability inside the final softmax layer (::TODO:: check resnet softmax)

            total += 1
            correct_predictions += (predicted == labels).sum().item()
            testbar.update(1)
    
    accuracy = correct_predictions/total

    print(f'Testing completed, accuracy: {accuracy}')


                
                


def taskb ():
    I = 2 



def taskc(resnet, input_channels):
    #Conv2d convert the image from 13 channels to 3 channels in output, using a kernel_size 3x3, letting the image factor to 1(stride = 1), with padding = 1(kernel_size/2)
    preprocess_layer = nn.Conv2d(input_channels, 3, kernel_size = 3, stride = 1, padding = 1)
    #Add preprocess layer to resnet 50 
    model = nn.Sequential(preprocess_layer, resnet)
    return model
    


if __name__ == '__main__':
    main()