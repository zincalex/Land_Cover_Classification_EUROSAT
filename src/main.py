# TODO chiedere al prof quale sia migliore/convenzione tra patterns, instances, samples, etc....
# TODO chiedere al prof perchè la prima run vada così lenta nel buildare il dataset
# TODO come è possibile che nel paper hanno fatto analisi resnet su solo 2 bande o 4 bande
# TODO introdurre possibile salvataggio del dataset 13 canali in file csv
# dimensionality reduction, PCA

# All resnet50 pretrained models require 3 x 224 x 224 sizes

# based 
import os               # paths
import tifffile  
import argparse
import numpy as np      # arrays 
import matplotlib.pyplot as plt 
import torch.utils
from tqdm import tqdm   # progress bar

# torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from torchvision.models import resnet50, ResNet50_Weights



parser = argparse.ArgumentParser()
parser.add_argument("-t", type = int, help="Analysis type", default = 0)
args = parser.parse_args()

ANALYSIS = args.t
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
torch.set_default_device(DEVICE)
if torch.cuda.is_available():
    g_device = torch.Generator(device='cuda')
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
        self.labels = labels                     # classes
        self.instances = instances               # images
        self.transform = transform               # transformations

    def __len__(self) :
        return len(self.instances)
    
    def __getitem__(self, idx) :
        return self.transform(self.instances[idx]), self.labels[idx] 

    def __getchannelesvalue__(self, bands_selected) :
        return self.transform(self.instances[:,:,bands_selected]), self.labels[:]


class EncoderDecoderCNN(nn.Module) :
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(13, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential (
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



def load_tif_channels(img_path, bands_selected):
    file = tifffile.imread(img_path)
    channels = file[..., bands_selected]
    channels = channels.astype(np.float32)/65535.0
    return channels


def load_data(img_path):
    """Read the an image through the tifffile library and convert the values
       of the image from uint16 to float32 

    Args:
        img_path: the path to a .tif image

    Returns:
        The image as numpy array
    
    """
    img_np_array = tifffile.imread(img_path)
    return img_np_array.astype(np.float32)


def save_model_parameters(model, model_parameters_path, model_parameters_name):
    """Save the parameters of the model inside the specified path in a .pth file

    Args:
        model:                     type of NN used
        model_parameters_path:     filesystem location and name of the file to be saved
    """
    print("\nSaving model...")
    if not os.path.exists(model_parameters_path) :
        os.makedirs(model_parameters_path)
        print("sgrodo")
    torch.save(model.state_dict(), model_parameters_path + '/' + model_parameters_name)
    print(f'MODEL {model_parameters_name} SAVED to {model_parameters_path}')




def resnet50_training(model, train_data, lf, optimizer, epochs, bands_name):
    """

    """
    model_parameters_path = '../parameters'
    model_parameters_name = f'{bands_name}.pth'
    for epoch in range(epochs):    
        with tqdm(total=len(train_data), unit='instance') as inpbar:
            for data in train_data:
                images, labels = data[0].to(DEVICE), data[1].to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = lf(outputs, labels)

                loss.backward()  # Backward pass
                optimizer.step()
                inpbar.update(1)
        print(f'Training loss: {loss.item()}          epoch: {epoch}\n')
    
    save_model_parameters(model, model_parameters_path, model_parameters_name)





def main () : 

    # PARAMETERS
    input_size = (64, 64)                                               # image size
    bands = 13                                                          # number of channels
    fraction_train = 0.8                                                # training split
    fraction_test = 1 - fraction_train                                  # test split
    batch_size = 32                                                     # batch size
    lr = 1e-4                                                           # learning rate
    factor = 20                                                         # learning rate factor for tuning
    epochs = 1                                                          # fixed number of epochs
    subset_bands = [[3,2,1], [0, 8, 9]]
    subset_names = ['RGB', 'Atmosperic_Factors']
    pretrained_flag = 0

    # DATASET
    dataset_path = '../dataset/'
    if(os.path.isdir(dataset_path)):
        labels = os.listdir(dataset_path) 
    num_classes = len(labels)                   # number of classes, 10 for the EuroSAT dataset

    dict_class = {}
    for i in range(len(labels)) : 
        dict_class.update({labels[i] : i + 1})

    print("CREATING DATASET...")
    train_instances, train_label, test_instances, test_label = [], [], [], []
    with tqdm(total=num_classes,  unit='label') as pbar:
        for label in os.listdir(dataset_path) :
            label_dir = os.path.join(dataset_path, label)
            images = os.listdir(label_dir)

            m_training = int(len(images) * fraction_train)          # Training set
            for i in range(m_training) : 
                img_path = label_dir + '/' + images[i]
                train_instances.append(load_data(img_path))
                train_label.append(dict_class[label])

            for i in range(m_training, len(images)) :               # Test set
                img_path = label_dir + '/' + images[i]
                test_instances.append(load_data(img_path))
                test_label.append(dict_class[label])
            pbar.update(1)

    transform = transforms.Compose([        # Define transformations to apply to the images (e.g., resizing, normalization)
        transforms.ToPILImage(),
        transforms.Resize(256),             # Resize images to 256x256                                                          
        transforms.CenterCrop(224),         # Crop the images 224x224, required by resnet50
        transforms.ToTensor(),              # Convert images to PyTorch tensors (standardization automatically applied)                                                                                                                                                                           
        #transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])                        # Normalize images
    ])
    
    # Main dataset, 13 channels
    train_dataset = EuroSATDataset(np.array(train_instances), train_label, transform)  
    test_dataset = EuroSATDataset(np.array(test_instances), test_label, transform)  

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, generator = g_device)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True, generator = g_device)

    print("DATASET CREATION COMPLETE\n\n")
    

    if (ANALYSIS == 1) : 
        print("Starting analysis: different resnet50s are trained on different band combinations; the outputs are weighted before producing an output")

        print("Creating dataset subsets...")
        # Subdataset with lesser channels
        train_dataset_RGB = EuroSATDataset(np.array(train_instances)[:,:,:,subset_bands[0]], train_label, transform)
        train_dataset_atmosferic = EuroSATDataset(np.array(train_instances)[:,:,:,subset_bands[1]], train_label, transform)  


        RGB_channels_train_data_loader = torch.utils.data.DataLoader(train_dataset_RGB, batch_size = batch_size, shuffle = True, generator = g_device)
        atmos_channels_train_data_loader = torch.utils.data.DataLoader(train_dataset_atmosferic, batch_size = batch_size, shuffle = True, generator = g_device)
        print("Complete")
        
        for i, sub_bands in enumerate(subset_bands) :
            model = resnet50(weights = None)
            model.to(DEVICE)
            
            loss_funct = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)         #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

            print(f"Starting Training on resnet50 number {i+1} on {subset_names[i]} bands")
            resnet50_training(model=model, train_data = RGB_channels_train_data_loader, lf=loss_funct, optimizer=optimizer, epochs=epochs, bands_name=subset_names[i])


    elif( ANALYSIS == 2) : 
        print("Starting analysis: layers are added before resnet50 for channel reduction from 13 to 3")
    elif( ANALYSIS == 3) :
        print("Starting analysis: encoder-decoder structure for channel reduction on the dataset. Resnet50 is then trained")
    else :
        print("suca")
        exit(1)

    
    '''if(pretrained_flag == 0):
        
    else:
        model.load_state_dict(torch.load(pretrained_model_path, map_location=DEVICE))
        print("Pretrained model has been loaded")
    '''

    model.eval()
    total = 0
    correct_predictions = 0

    print("\nTESTING: START")
    with tqdm(total=len(test_data_loader), unit='instance') as testbar:
        for i, test_data in enumerate(test_data_loader):
            images, labels = test_data[0].to(DEVICE), test_data[1].to(DEVICE)
            outputs = model(images)

            _,predicted = torch.max(outputs, 1) #max 1 probability, takes the max probability inside the final softmax layer (::TODO:: check resnet softmax)

            total += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            testbar.update(1)
    
    accuracy = correct_predictions/total 

    print(f'TESTING : DONE')
    print(F'Accuracy = {accuracy}')

                 

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