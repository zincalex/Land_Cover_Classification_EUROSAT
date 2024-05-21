# TODO rendere la parte di building del dataset opzionale
# TODO showare la confusion matrxi per task 1
# TODO togliere il wall of text di dataset ma metterli dentro il for
# TODO cercare di risolvere il problema del transform con la normalizzazione
# dimensionality reduction, PCA
# KEYWORD ensemble, slides pg367 (high chance of a 1-2% improving)
# KEYWORD bagging and meta classifier, should be compatible with resnet. dataset must have the same size and can have same elements
# finish looking at slide n373


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
from torch.utils.data import Dataset, random_split
from torchvision import transforms, datasets
from torchvision.models import resnet50, ResNet50_Weights

# sklearn
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score


parser = argparse.ArgumentParser()
parser.add_argument("-t", type = int, help="Analysis type", default = 0)
args = parser.parse_args()

SKIP_DATASET_CREATION = True
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



verita = []
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
        return self.transform(self.instances[idx]), torch.tensor(self.labels[idx] , dtype=torch.long)
    
    def __getlabels__(self) : 
        return self.labels

    def __getchannelesvalue__(self, bands_selected) :
        return self.transform(self.instances[:,:,bands_selected]), self.labels[:]


def load_img(img_path):
    """Read the an image through the tifffile library and convert the values
       of the image from uint16 to float32 

    Args:
        img_path: the path to a .tif image

    Returns:
        The image as numpy array
    
    """
    img_np_array = tifffile.imread(img_path)
    return img_np_array.astype(np.float32)


def save_to_hdf5(train_instances, test_instances, directory_name, file_name, train_label = None, test_label = None) :
    if not os.path.exists(f'../{directory_name}') :
        os.makedirs(f'../{directory_name}')

    with h5py.File(f'../{directory_name}/{file_name}', 'w') as hf:      # Creation of .h5 file ----> compression lossless
        # Create a group for train data
        train_group = hf.create_group('train')
        train_group.create_dataset('data', data = train_instances)
        if train_label != None :
            train_group.create_dataset('labels', data=train_label)  

        # Create a group for test data
        test_group = hf.create_group('test')
        test_group.create_dataset('data', data = test_instances)
        if test_label != None :
            test_group.create_dataset('labels', data=test_label) 


def load_hdf5_PCA(file_path) : 
    with h5py.File(f'{file_path}', 'r') as hf :
            train_instances = hf['train']['data'][:]
            test_instances = hf['test']['data'][:]

    return train_instances, test_instances


def load_hdf5_EUROSAT(file_path) : 
    with h5py.File(f'{file_path}', 'r') as hf :
            train_instances = hf['train']['data'][:]
            train_labels = hf['train']['labels'][:]
            test_instances = hf['test']['data'][:]
            test_labels = hf['test']['labels'][:]

    return train_instances, train_labels, test_instances, test_labels


def save_model_parameters(model, model_parameters_path, model_parameters_name):
    """Save the parameters of the model inside the specified path in a .pth file

    Args:
        model:                     type of NN used
        model_parameters_path:     filesystem location and name of the file to be saved
    """
    print("\nSaving model...")
    if not os.path.exists(model_parameters_path) :
        os.makedirs(model_parameters_path)
    torch.save(model.state_dict(), model_parameters_path + '/' + model_parameters_name)
    print(f'MODEL {model_parameters_name} SAVED to {model_parameters_path}')


def calculate_mean_std(train_instances, test_instances, bands) :
    """Calculate per each band the mean and the standard deviation

    Args:
        train_instances:    np.array filled with images used for training                  
        test_instances:     np.array filled with images used for testing
        bands:              number of channels of the images, which have shapes (64x64xbands)     
    """
    mean = np.zeros(bands)
    std = np.zeros(bands)
    dataset = np.concatenate((train_instances, test_instances), axis=0)
    for i in range(bands) :
        mean[i] = np.mean(dataset[:,:,:,i])
        std[i] = np.std(dataset[:,:,:,i])

    return mean.tolist(), std.tolist()


def create_EuroSATDatasets(instances, labels, subset_bands, transform) :
    datasets = []
    for bands in subset_bands:
        dataset = EuroSATDataset(instances[:, :, :, bands], labels, transform)
        datasets.append(dataset)
    return datasets


def train_val_split_datasets(train_datasets, fraction_train, generator) : 
    split_datasets = []
    for dataset in train_datasets:
        num_data = len(dataset)
        num_train = int(fraction_train * num_data)
        num_val = num_data - num_train
        train_dataset, val_dataset =  torch.utils.data.random_split(dataset, lengths=[num_train, num_val], generator=generator)
        split_datasets.append((train_dataset, val_dataset))
    return split_datasets


def create_data_loaders(train_datasets, val_datasets, test_datasets, batch_size, generator):
    train_loaders = [torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator) for dataset in train_datasets]
    val_loaders = [torch.utils.data.DataLoader(dataset, batch_size=batch_size) for dataset in val_datasets]
    test_loaders = [torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True) for dataset in test_datasets]
    return train_loaders, val_loaders, test_loaders


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


def resnet50_test(model, data_loader) :
    total = 0
    correct_predictions = 0
    predictions = []
    true_labels = []
    with tqdm(total=len(data_loader), unit='instance') as testbar:
        for test_data in data_loader :
            images, labels = test_data[0].to(DEVICE), test_data[1].to(DEVICE)
            
            with torch.no_grad() :
                outputs = model(images)
                probabilities = nn.functional.softmax(outputs, dim=1)    # Softmax layer
                predicted = torch.argmax(probabilities, 1) 

            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            verita.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            testbar.update(1)
    
    accuracy = correct_predictions / total
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    return accuracy, precision, recall, f1, np.array(predictions), np.array(true_labels)

def to_one_hot(predictions, num_classes) :
    predictions_tensor = torch.tensor(predictions, dtype=torch.long)
    one_hot = nn.functional.one_hot(predictions_tensor, num_classes=num_classes)
    return one_hot.cpu().numpy()


def resnet50_majority_voting_labels(model_list, model_weights, test_loader_list):
    predictions = []
    for test_loader, model, weight in zip(test_loader_list, model_list, model_weights):
        ensemble_preds = []

        for images, _ in test_loader:
            images = images.to(DEVICE)

            model.eval()
            model.to(DEVICE)

            with torch.no_grad():
                outputs = model(images)
                probabilities = nn.functional.softmax(outputs, dim=1)
                predicted_label = torch.argmax(probabilities, dim=1)

            ensemble_preds.append(predicted_label*weight) 
    
    final_predictions = torch.sum(torch.stack(ensemble_preds), dim=0)
    vote = torch.argmax(final_predictions)
    predictions.append(vote.item())
    print(len(predictions))
    return predictions


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
    epochs = 1                                                         # fixed number of epochs
    subset_bands = [[3,2,1], [0, 8, 9], [4,5,6], [7,11,12]]
    subset_names = ['RGB', 'Atmosperic_Factors', 'Red_Edge', 'SWIR']

    # DATASET
    dataset_path = '../dataset/'
    if(os.path.isdir(dataset_path)):
        labels = os.listdir(dataset_path) 
    num_classes = len(labels)                   # number of classes, 10 for the EuroSAT dataset

    dict_class = {}
    for i in range(len(labels)) : 
        dict_class.update({labels[i] : i})

    print("DATA PRE-PROCESSING")
    train_instances, train_label, test_instances, test_label = [], [], [], []
    if not SKIP_DATASET_CREATION :
        print("Loading images from EUROSAT dataset")
        with tqdm(total=num_classes,  unit='class') as pbar:
            for label in os.listdir(dataset_path) :
                label_dir = os.path.join(dataset_path, label)
                images = os.listdir(label_dir)

                m_training = int(len(images) * fraction_train)          # Training set
                for i in range(m_training) : 
                    img_path = label_dir + '/' + images[i]
                    train_instances.append(load_img(img_path))
                    train_label.append(dict_class[label])

                for i in range(m_training, len(images)) :               # Test set
                    img_path = label_dir + '/' + images[i]
                    test_instances.append(load_img(img_path))
                    test_label.append(dict_class[label])
                pbar.update(1)

        train_instances = np.array(train_instances)
        test_instances = np.array(test_instances)
        print('Saving EUROSAT iamges as numpy arrays')
        save_to_hdf5(train_instances, test_instances, directory_name="EUROSAT_numpy", file_name="imgs_numpy.h5", train_label=train_label, test_label=test_label)
        print('Done')
    else : 
        print('Loading pre-processed EUROSAT dataset as numpy arrays')
        train_instances, train_label, test_instances, test_label = load_hdf5_EUROSAT(file_path="../EUROSAT_numpy/imgs_numpy.h5")
        print("Done")

    num_train_instances = len(train_instances)
    num_test_instances = len(test_instances)
    print('Calculating mean and standard deviation for each of the 13 channels')   
    #mean, std = calculate_mean_std(train_instances, test_instances, bands) 
    print("Done")

    transform = transforms.Compose([        # Define transformations to apply to the images (e.g., resizing, normalization)
        transforms.ToPILImage(),
        transforms.Resize(256),             # Resize images to 256x256                                                          
        transforms.CenterCrop(224),         # Crop the images 224x224, required by resnet50
        transforms.ToTensor(),              # Convert images to PyTorch tensors (standardization automatically applied)                                                                                                                                                                           
        #transforms.Normalize(mean = mean, std = std)                        # Normalize images
    ])
    
    # Main dataset, 13 channels
    train_dataset = EuroSATDataset(train_instances, train_label, transform) 
    test_dataset = EuroSATDataset(test_instances, test_label, transform)  

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, generator = g_device)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True, generator = g_device)
    print("DATA PRE-PROCESSING COMPLETE\n\n")
    
    
    if (ANALYSIS == 1) : 
        print("Starting analysis: different resnet50s are trained on different band combinations; the outputs are weighted before producing an output")

        print("Creating dataset subsets...")
        # Sub dataset with lesser channels
        train_datasets_subchannels = create_EuroSATDatasets(train_instances, train_label, subset_bands, transform)
        test_datasets_subchannels = create_EuroSATDatasets(test_instances, test_label, subset_bands, transform)

        # Train validation split of the images
        train_val_datasets = train_val_split_datasets(train_datasets_subchannels, fraction_train, generator = g_device)
        train_datasets_subchannels, val_datasets_subchannels = zip(*train_val_datasets)

        # Creating data loaders
        dataloader_train_list, dataloader_val_list, dataloader_test_list= create_data_loaders(train_datasets_subchannels, val_datasets_subchannels, test_datasets_subchannels, batch_size, g_device)
        print("Complete")
        

        # TRAINING each model individually
        model_parameters_path = '../parameters'
        model_list = []
        for i, sub_bands in enumerate(subset_bands) :
            if i == 0 : 
                model = resnet50(weights = ResNet50_Weights.DEFAULT)
            else :
                model = resnet50(weights = None)

            num_features = model.fc.in_features     # number of features in input in the last FC layer
            model.fc = torch.nn.Linear(num_features, num_classes)
            model.to(DEVICE)
            
            loss_funct = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)         #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

            model_parameters_name = f'{subset_names[i]}.pth'
            print(f"Starting Training on resnet50 number {i+1} on {subset_names[i]} band")
            resnet50_training(model, dataloader_train_list[i], loss_funct, optimizer, epochs)
            save_model_parameters(model, model_parameters_path, model_parameters_name)

            model_list.append(model) # All the net is locally stored (shallow copy) inside the list
        

        # VALIDATION
        # We use the validation set to determine the weights used for the ensembling of the resnet's (the weight w_i is the accuracy of the resnet i)
        print("\nValidation: start") 
        ensamble_weights = []      
        with tqdm(total=len(model_list), unit='models') as valbar:
            for i, model in enumerate(model_list) :
                model.eval()
                results = resnet50_test(model, dataloader_val_list[i]) 
                ensamble_weights.append(results[0])  # Taking only the accuracy from the method resnet50_test
                valbar.update(1)
        print("\nValidation: done") 


        # TESTING
        print("\nTesting: start") 
        all_classifiers_predictions = []     
        with tqdm(total=len(model_list), unit='models') as testbar:

            # First testing all models and transforming the class predictions in one hot encoding
            for i, model in enumerate(model_list) :
                model.eval()
                accuracy, precision, recall, f1, predictions, true_labels = resnet50_test(model, dataloader_test_list[i])
                predictions = to_one_hot(predictions, num_classes)
                all_classifiers_predictions.append(predictions)     # list of numpy arrays
                testbar.update(1)

        correct_pred = 0
        num_images = len(all_classifiers_predictions[0])
        predict_confusion_matrix = []
        with tqdm(total=num_images, unit='img') as bar:
            for i in range(num_images) : # For each image
                
                one_hot_weighted = []
                for j in range(len(model_list)) :   # For each classifier
                    print(f'predizione modello {j} su immagine {i}: {all_classifiers_predictions[j][i]}')
                    one_hot_weighted.append(all_classifiers_predictions[j][i] * ensamble_weights[j])
                
                one_hot_weighted = np.array(one_hot_weighted)
                result = np.sum(one_hot_weighted, axis=0) # Sum element-wise along the columns
                net_prediction = np.argmax(result)

                predict_confusion_matrix.append(net_prediction)
                if net_prediction == true_labels[i] :
                    correct_pred += 1
                bar.update(1)

        accuracy = correct_pred / num_images
        precision = precision_score(true_labels, predict_confusion_matrix, average='weighted')
        recall = recall_score(true_labels, predict_confusion_matrix, average='weighted')
        f1 = f1_score(true_labels, predict_confusion_matrix, average='weighted')

        print(f'Accuracy = {accuracy}')
        print(f'Precision = {precision}')
        print(f'Recall = {recall}')
        print(f'F1 Score = {f1}')
        show_confusion_matrix(predict_confusion_matrix, true_labels)
        print("\nTesting: done") 

    elif(ANALYSIS == 2) : 
        print("Starting analysis: PCA layer is added before resnet50 for channel reduction from 13 to 3")

        if not SKIP_PCA :  
            # Shape of train_instances (num_instances, 64, 64, 13) ----> (num_samples, 64*64, 13) 
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
            save_to_hdf5(reconstructed_test_imgs, reconstructed_test_imgs, directory_name="PCA_dataset", file_name="imagesPCA.h5")
            print('Done')

        else : # PCA computation already done
            print('Loading modified PCA dataset...')
            reconstructed_train_imgs, reconstructed_test_imgs = load_hdf5_PCA(file_path="../PCA_dataset/imagesPCA.h5")
            print("Done")

        mean, std = calculate_mean_std(reconstructed_train_imgs, reconstructed_test_imgs, bands = 3) 
        transform = transforms.Compose([                            # Define transformations to apply to the images (e.g., resizing, normalization)
            transforms.ToPILImage(),
            transforms.Resize(256),                                 # Resize images to 256x256                                                          
            transforms.CenterCrop(224),                             # Crop the images 224x224, required by resnet50
            transforms.ToTensor(),                                  # Convert images to PyTorch tensors (standardization automatically applied)                                                                                                                                                                           
            transforms.Normalize(mean = mean, std = std)            # Normalize images
        ])

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
        accuracy, precision, recall, f1, predictions, correct_labels = resnet50_test(model, test_data_loader)   #loss_funct
        print(F'Accuracy = {accuracy}')
        print(f'Precision = {precision}')
        print(f'Recall = {recall}')
        print(f'F1 Score = {f1}')
        show_confusion_matrix(correct_labels, predictions)

    elif( ANALYSIS == 3) :
        print("Starting analysis: encoder-decoder structure for channel reduction on the dataset. Resnet50 is then trained")
    else :
        print("suca")
        exit(1)


if __name__ == '__main__':
    main()