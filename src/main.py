# TODO chiedere al prof quale sia migliore/convenzione tra patterns, instances, samples, etc....


# All resnet50 pretrained models require 3 x 224 x 224 sizes
# Visto che il data set è diviso in cartelle, noi facciamo che prendiamo una cartella e prendiamo x% delle prime immagini
# Per ogni classe, creare le matrici di training con valore + label -----> assegnare noi le label
# fare shuffle una volta preso tutto il training set

# based 
import os               # paths
import numpy as np      # arrays 
import matplotlib.pyplot as plt 
from PIL import Image
#from osgeo import gdal  from tif to rgb
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
        self.labels = labels  
        self.instances = instances                               # images
        self.transform = transform                               # transforms

    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        return self.transform(self.instances[idx]), self.labels[idx] 





def main () : 

    # PARAMETERS
    input_size = (64, 64)                       # image size
    bands = 13                                  # number of channels
    fraction_train = 0.8                        # training split
    fraction_test = 1 - fraction_train          # test split
    batch_size = 32                             # batch size
    lr = 1e-4                                   # learning rate
    factor = 20                                 # learning rate factor for tuning
    epochs = 5                                  # fixed number of epochs


    # DATASET
    dataset_path = '../dataset/'
    labels = os.listdir(dataset_path) 
    dict_class = {}
    for i in range(len(labels)) : 
        dict_class.update({labels[i] : i + 1})

    num_classes = len(labels)                   # number of classes, 10 for the EuroSAT dataset
    

    train_instances, train_label, test_instances, test_label = [], [], [], []
    for label in os.listdir(dataset_path) :
        label_dir = os.path.join(dataset_path, label)
        images = os.listdir(label_dir)

        m_training = int(len(images) * fraction_train)
        m_test = int(len(images) * fraction_test)
        
        for i in range(m_training) : 
            img_path = label_dir + '/' + images[i]
            tif_img = Image.open(img_path)
            train_instances.append(np.array(tif_img))
            train_label.append(dict_class[label])

        for i in range(m_training, len(images)) :               # the remaining images are for the test set 
            img_path = label_dir + '/' + images[i]
            tif_img = Image.open(img_path)
            test_instances.append(np.array(tif_img))
            test_label.append(dict_class[label])

        
    # Define transformations to apply to the images (e.g., resizing, normalization)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),           # Convert images to PyTorch tensors
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
    ])
    
    

    train_dataset = EuroSATDataset(train_instances, train_label, transform)  
    test_dataset = EuroSATDataset(test_instances, test_label, transform)  

    
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True)
    
    
    print(train_dataset.__getitem__(500))
    
    

    # RESNET50
    model = resnet50(weights = ResNet50_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(bands, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    loss_funct = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.to(DEVICE)
    with tqdm(total=epochs, unit='epoch') as pbar:
        for epoch in range(epochs):
            model.train()
            for images, labels in train_data_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model(images)
                loss = loss_funct(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print('Training loss: ', loss.item(), 'epoch: ', epoch)
            pbar.update(1)
                
                
            






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