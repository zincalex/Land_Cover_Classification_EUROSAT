# All resnet50 pretrained models require 3 x 224 x 224 sizes
# Visto che il data set è diviso in cartelle, noi facciamo che prendiamo una cartella e prendiamo x% delle prime immagini

# based 
import os               # paths
import numpy as np      # arrays 
#from osgeo import gdal  
from tqdm import tqdm   # progress bar

# torch
import torch
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



def main () : 
    folders_path = '../dataset/'
    classes = os.listdir(folders_path) 
    print(classes)


    resnet50(weights=ResNet50_Weights.DEFAULT)



if __name__ == '__main__':
    main()