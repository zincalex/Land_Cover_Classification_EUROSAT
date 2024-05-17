# TODO chiedere al prof quale sia migliore/convenzione tra patterns, instances, samples, etc....
# TODO chiedere al prof perchè la prima run vada così lenta nel buildare il dataset
# TODO come è possibile che nel paper hanno fatto analisi resnet su solo 2 bande o 4 bande
# TODO introdurre possibile salvataggio del dataset 13 canali in file csv
# dimensionality reduction, PCA
# KEYWORD ensemble, slides pg367 (high chance of a 1-2% improving)
# KEYWORD bagging and meta classifier, should be compatible with resnet. dataset must have the same size and can have same elements
# finish looking at slide n373


# All resnet50 pretrained models require 3 x 224 x 224 sizes

# based 
import os               # paths
import tifffile  
import argparse
import numpy as np      # arrays 
import matplotlib.pyplot as plt 
import torch.utils
import h5py             # compression of large dataset and upload
from tqdm import tqdm   # progress bar

# torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from torchvision.models import resnet50, ResNet50_Weights

# sklearn
from sklearn.decomposition import PCA
from sklearn import metrics


parser = argparse.ArgumentParser()
parser.add_argument("-t", type = int, help="Analysis type", default = 0)
args = parser.parse_args()

SKIP_PCA = True
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
    
    def __getlabels__(self) : 
        return self.labels

    def __getchannelesvalue__(self, bands_selected) :
        return self.transform(self.instances[:,:,bands_selected]), self.labels[:]

#######Attenzione: da verificare l'uso di ConvTranspose2d e Conv2 in quanto dovrebbe essere inverso(Conv2d per diminuire il numero di Canali e ConvTranspose2d per aumentarli)
# https://stackoverflow.com/questions/68976745/in-keras-what-is-the-difference-between-conv2dtranspose-and-conv2d qui fa riferimento a keras ma il discorso dovrebbe essere lo stesso, confermate???

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


def resnet50_training(model, train_data_loader, lf, optimizer, epochs):
    """Train a model with the specified hyperparameters

    Args:
        model:                      model to train
        train_data_loader:          training EuroSATDataset (img, label)
        lf:                         loss function
        optimizer:                  optimizer used in the backward pass 
        epochs:                     number of epochs for training
    """
    for epoch in range(epochs):    
        with tqdm(total=len(train_data_loader), unit='instance') as inpbar:
            for data in train_data_loader :
                images, labels = data[0].to(DEVICE), data[1].to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = lf(outputs, labels)

                loss.backward()  # Backward pass
                optimizer.step()
                inpbar.update(1)
        print(f'Training loss: {loss.item()}          epoch: {epoch}\n')


def resnet50_test(model, test_data_loader, lf) :
    total = 0
    correct_predictions = 0
    predictions = []
    correct_labels = []
    with tqdm(total=len(test_data_loader), unit='instance') as testbar:
        for test_data in test_data_loader :
            images, labels = test_data[0].to(DEVICE), test_data[1].to(DEVICE)
            outputs = model(images)
            
            probabilities = nn.functional.softmax(outputs, dim=1)    # Softmax layer
            _,predicted = torch.max(probabilities, 1) 

            predictions.extend(predicted.cpu().numpy())
            correct_labels.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            testbar.update(1)
    
    accuracy = correct_predictions/total
    return accuracy, predictions, correct_labels


def show_confusion_matrix(correct_labels, predictions, title = "Confusion Matrix") : 
    disp = metrics.ConfusionMatrixDisplay.from_predictions(correct_labels, predictions)
    disp.figure_.suptitle(title)
    print(f"Confusion matrix: \n{disp.confusion_matrix}")
    plt.tight_layout()
    plt.show()








def main () : 

    # PARAMETERS
    input_size = (64, 64)                                               # image size
    bands = 13                                                          # number of channels
    fraction_train = 0.8                                                # training split
    fraction_test = 1 - fraction_train                                  # test split
    batch_size = 32                                                     # batch size
    lr = 1e-4                                                           # learning rate
    factor = 20                                                         # learning rate factor for tuning
    epochs = 5                                                         # fixed number of epochs
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
        dict_class.update({labels[i] : i})

    print("CREATING DATASET...")
    train_instances, train_label, validation_instances, validation_labels, test_instances, test_label = [], [], [], [], [], []
    with tqdm(total=num_classes,  unit='class') as pbar:
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
    
    train_instances = np.array(train_instances)
    test_instances = np.array(test_instances)

    # Main dataset, 13 channels
    train_dataset = EuroSATDataset(train_instances, train_label, transform)
    #validation_dataset = EuroSATDataset  
    test_dataset = EuroSATDataset(test_instances, test_label, transform)  

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
        
        model_parameters_path = '../parameters'
        for i, sub_bands in enumerate(subset_bands) :
            model = resnet50(weights = None)
            num_features = model.fc.in_features     # number of features in input in the last FC layer
            model.fc = torch.nn.Linear(num_features, num_classes)
            model.to(DEVICE)
            
            loss_funct = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)         #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

            model_parameters_name = f'{subset_names[i]}.pth'
            print(f"Starting Training on resnet50 number {i+1} on {subset_names[i]} bands")
            resnet50_training(model=model, train_data = RGB_channels_train_data_loader, lf=loss_funct, optimizer=optimizer, epochs=epochs)
            save_model_parameters(model, model_parameters_path, model_parameters_name)


    elif(ANALYSIS == 2) : 
        print("Starting analysis: PCA layer is added before resnet50 for channel reduction from 13 to 3")

        if not SKIP_PCA :  
            # Shape of train_instances (num_instances, 64, 64, 13) ----> (num_samples, 64*64, 13) 
            num_train_instances = len(train_instances)
            num_test_instances = len(test_instances)

            flattened_train_instances = train_instances.reshape(num_train_instances, -1, 13)
            flattened_test_instances = test_instances.reshape(num_test_instances, -1, 13)

            pca = PCA(n_components = 3)
            transformed_train_imgs = []
            transformed_test_imgs = []
            
            print("PCA transformation")
            with tqdm(total=(num_train_instances + num_test_instances), unit='img') as pbar:
                for img in flattened_train_instances : 
                    transformed_train_imgs.append(pca.fit_transform(img))
                    pbar.update(1)
                
                for img in flattened_test_instances : 
                    transformed_test_imgs.append(pca.fit_transform(img))
                    pbar.update(1)

            transformed_train_imgs = np.array(transformed_train_imgs)
            reconstructed_train_imgs = transformed_train_imgs.reshape(num_train_instances, 64, 64, 3)

            transformed_test_imgs = np.array(transformed_test_imgs)
            reconstructed_test_imgs = transformed_test_imgs.reshape(num_test_instances, 64, 64, 3)

            print('Saving modified dataset...')
            if not os.path.exists('../PCA_dataset') :
                os.makedirs('../PCA_dataset')
            with h5py.File('../PCA_dataset/imagesPCA.h5', 'w') as hf:      # Creation of .h5 file ----> compression lossless
                # Create a group for train data
                train_group = hf.create_group('train')
                train_group.create_dataset('data', data = reconstructed_train_imgs)

                # Create a group for test data
                test_group = hf.create_group('test')
                test_group.create_dataset('data', data = reconstructed_test_imgs)
            print('Done')
        
        else : # PCA computation already done
            print('Loading modified PCA dataset...')
            with h5py.File('../PCA_dataset/imagesPCA.h5', 'r') as hf :
                reconstructed_train_imgs= hf['train']['data'][:]
                reconstructed_test_imgs = hf['test']['data'][:]
            print("Done")

        # Start computation, on the channel reduced dataset through PCA
        train_dataset_PCA = EuroSATDataset(reconstructed_train_imgs, train_label, transform)
        test_dataset_PCA = EuroSATDataset(reconstructed_test_imgs, test_label, transform)  

        train_data_loader = torch.utils.data.DataLoader(train_dataset_PCA, batch_size = batch_size, shuffle = True, generator = g_device)
        test_data_loader = torch.utils.data.DataLoader(test_dataset_PCA, batch_size = batch_size, shuffle = True, generator = g_device)


        print("\nTraining: start")    
        model = resnet50(weights = ResNet50_Weights.DEFAULT)
        num_features = model.fc.in_features     # number of features in input in the last FC layer
        model.fc = torch.nn.Linear(num_features, num_classes)
        model.to(DEVICE)
        
        loss_funct = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  
        resnet50_training(model, train_data_loader, loss_funct, optimizer, epochs)

        model.eval()
        print("\nTesting: start")
        accuracy, predictions, correct_labels = resnet50_test(model, test_data_loader, loss_funct)
        print(F'Accuracy = {accuracy}')
        show_confusion_matrix(correct_labels, predictions)



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
    """ 
    
    """
                 

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