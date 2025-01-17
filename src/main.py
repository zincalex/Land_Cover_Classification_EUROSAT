# Alessandro Viespoli 2120824, Kabir Bertan 2122545, Francesco Colla 2122543
# Land Cover Classification using Sentinel-2 Satellite Images
# UniPD 2023/24 - Deep Learning Project

# Base 
import os                           # paths
import numpy as np                  # arrays 
import argparse                
from tqdm import tqdm               # progress bar

# Torch
import torch
import torch.utils
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import densenet121, DenseNet121_Weights, densenet161, DenseNet161_Weights

# Sklearn
from sklearn import metrics
from sklearn.decomposition import PCA

# Classes and Functions
import dataset, plots



parser = argparse.ArgumentParser()
parser.add_argument("-t", type = int, help="Analysis type", default = 1)
parser.add_argument("-d", type = int, help="Skip dataset creation", default = 0)
parser.add_argument("-p", type = int, help="Skip PCA dataset creation", default = 0)
args = parser.parse_args()

if args.t not in (0, 1, 2) :
    raise ValueError(f"Invalid value for -t: {args.t}. Expected values are 0 or 1 or 2.")
if args.d not in (0, 1) :
    raise ValueError(f"Invalid value for -t: {args.d}. Expected values are 0 or 1.")
if args.p not in (0, 1) :
    raise ValueError(f"Invalid value for -t: {args.p}. Expected values are 0 or 1.")

SKIP_DATASET_CREATION = args.d
SKIP_PCA = args.p
ANALYSIS = args.t
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
torch.set_default_device(DEVICE)
if torch.cuda.is_available():
    g_device = torch.Generator(device='cuda')
print(f"Using {DEVICE} device\n")



# FUNCTIONS 
def model_training(model, train_data_loader, lf, optimizer, epochs):
    """Train a resnet50 architecture (model) with the specified hyperparameters

    Args:
        model (torchvision.models):                             model to train
        train_data_loader (torch.utils.data.DataLoader):        
        lf (torch loss function):                               loss function
        optimizer (torch.optim):                                optimizer used in the backward pass 
        epochs:                                                 number of epochs for training

    Returns: 
        A list with the training loss for each epoch
    """

    # Scheduler setup
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_losses = []
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
        train_losses.append(loss.item())
        scheduler.step()
    return train_losses


def model_test(model, test_data_loader) :
    """Test resnet50 architecture with the given data
 
    Args:
        model (torchvision.models):                          model to test
        test_data_loader (torch.utils.data.DataLoader):      

    Returns:
        The accuracy of the testing, a numpy array with the predictions of the net,
        a numpy array with the true labels of the images tested
    """
    total = 0
    correct_predictions = 0
    predictions = []
    true_labels = []
    with tqdm(total=len(test_data_loader), unit='instance') as testbar:
        for test_data in test_data_loader :
            images, labels = test_data[0].to(DEVICE), test_data[1].to(DEVICE)
            
            with torch.no_grad() :
                outputs = model(images)
                probabilities = nn.functional.softmax(outputs, dim=1)    # Softmax layer
                predicted = torch.argmax(probabilities, 1) 

            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            testbar.update(1)
    
    accuracy = correct_predictions / total
    return accuracy, np.array(predictions), np.array(true_labels)


def save_model_parameters(model, model_parameters_path, model_parameters_name) :
    """Save the parameters of the model inside the specified path in a .pth file

    Args:
        model (torchvision.models):       type of NN used
        model_parameters_path (string):   filesystem location and name of the file to be saved
    """
    print("Saving model...")
    if not os.path.exists(model_parameters_path) :
        os.makedirs(model_parameters_path)
    torch.save(model.state_dict(), model_parameters_path + '/' + model_parameters_name)
    print(f'MODEL {model_parameters_name} SAVED to {model_parameters_path}')


def calculate_mean_std(train_instances, test_instances, bands) :
    """For each band, the mean and the standard deviation are computed.

    Args:
        train_instances (numpy array):    train images                 
        test_instances (numpy array):     test images
        bands (int):                      number of channels of the images  

    Returns: 
        Two lists with the mean and the std for each channel  
    """
    mean = np.zeros(bands)
    std = np.zeros(bands)
    dataset = np.concatenate((train_instances, test_instances), axis=0)
    for i in range(bands) :
        mean[i] = np.mean(dataset[:,:,:,i])
        std[i] = np.std(dataset[:,:,:,i])

    return mean.tolist(), std.tolist()


def train_val_split_datasets(train_datasets, fraction_train, generator) : 
    """The method split the content inside each EuroSATDataset object, contained in train_datasets, in training 
       and validation sets.

    Args:
        train_datasets (list):          list with EuroSATDataset objects
        fraction_train (float):         train validation split value in [0,1]
        generator (torch.Generator)

    Returns:
        A list of tuples of EuroSATDataset objects for each dataset given
    """
    split_datasets = []
    for dataset in train_datasets:
        num_data = len(dataset)
        num_train = int(fraction_train * num_data)
        num_val = num_data - num_train
        train_dataset, val_dataset =  torch.utils.data.random_split(dataset, lengths=[num_train, num_val], generator=generator)
        split_datasets.append((train_dataset, val_dataset))
    return split_datasets


def to_one_hot(predictions, num_classes) :
    """Transform the predictions to one-hot encoding based on the number of classes  
 
    Args:
        predictions (numpy array):  class predictions (for our purpose in range [0,9]) 
        num_classes (int):          number of total classes (10)

    Returns:
        A numpy array where each element is a one-hot encoding of a prediction
    """
    predictions_tensor = torch.tensor(predictions, dtype=torch.long)
    one_hot = nn.functional.one_hot(predictions_tensor, num_classes=num_classes)
    return one_hot.cpu().numpy()




def main () : 

    # PARAMETERS
    input_size = (64, 64)                                               # image size
    bands = 13                                                          # number of channels
    fraction_train = 0.8                                                # training split
    fraction_test = 1 - fraction_train                                  # test split
    batch_size = 32                                                     # batch size
    lr = 1e-4                                                           # learning rate
    epochs = 20                                                         # fixed number of epochs
    subset_bands = [[3,2,1], [0, 9, 10], [4,5,6], [7,11,12]]
    subset_names = ['RGB', 'Atmosperic_Factors', 'Red_Edge', 'SWIR']

    # DATASET
    dataset_path = '../dataset/'
    if(os.path.isdir(dataset_path)):
        labels = os.listdir(dataset_path) 
    num_classes = len(labels)                   # number of classes, 10 for the EuroSAT dataset

    dict_class = {}
    for i in range(len(labels)) : 
        dict_class.update({labels[i] : i})      # e.g ("Annual Crop" : 0, "Forest" : 1, ...)


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
                    train_instances.append(dataset.load_img(img_path))
                    train_label.append(dict_class[label])

                for i in range(m_training, len(images)) :               # Test set
                    img_path = label_dir + '/' + images[i]
                    test_instances.append(dataset.load_img(img_path))
                    test_label.append(dict_class[label])
                pbar.update(1)

        train_instances = np.array(train_instances)
        test_instances = np.array(test_instances)
        print('Saving EUROSAT iamges as numpy arrays')
        dataset.save_to_hdf5(train_instances, test_instances, directory_name="EUROSAT_numpy", file_name="imgs_numpy.h5", train_label=train_label, test_label=test_label)
    else : 
        print('Loading pre-processed EUROSAT dataset as numpy arrays')
        train_instances, train_label, test_instances, test_label = dataset.load_hdf5_EUROSAT(file_path="../EUROSAT_numpy/imgs_numpy.h5")

    print("DATA PRE-PROCESSING COMPLETE\n")
    num_train_instances = len(train_instances)
    num_test_instances = len(test_instances)


    # LAND COVER CLASSIFICATION
    if (ANALYSIS == 1) : # ENSEMBLE ANALYSIS
        transform = transforms.Compose([                            
            transforms.ToPILImage(), 
            transforms.Resize(256),                                                                                          
            transforms.CenterCrop(224),                             
            transforms.ToTensor(),          # standardization automatically applied                                                                                                                                                                                                                  
        ])

        print("STARTING ANALYSIS: Ensemble of resnet50s trained on different band combinations")
        print("Creating dataset subsets")
        # Sub dataset with lesser channels
        train_datasets_subchannels = dataset.create_EuroSATDatasets(train_instances, train_label, subset_bands, transform)
        test_datasets_subchannels = dataset.create_EuroSATDatasets(test_instances, test_label, subset_bands, transform)

        # Train validation split of the images
        train_val_datasets = train_val_split_datasets(train_datasets_subchannels, fraction_train, generator = g_device)
        train_datasets_subchannels, val_datasets_subchannels = zip(*train_val_datasets)

        # Creating data loaders
        dataloader_train_list, dataloader_val_list, dataloader_test_list= dataset.create_data_loaders(train_datasets_subchannels, val_datasets_subchannels, test_datasets_subchannels, batch_size, g_device)
    
        
        # TRAINING each model individually
        model_parameters_path = '../parameters'
        model_list = []
        ensemble_losses = []
        for i, sub_band in enumerate(subset_bands) :
            # DenseNet structure
            model = densenet161(DenseNet161_Weights.DEFAULT)
            num_features = model.classifier.in_features     # number of features in input in the last FC layer
            model.classifier = torch.nn.Linear(num_features, num_classes)

            # ResNet structure
            #model = resnet50(weights = ResNet50_Weights.DEFAULT)
            #num_features = model.fc.in_features     # number of features in input in the last FC layer
            #model.fc = torch.nn.Linear(num_features, num_classes)

            model.to(DEVICE)
            loss_funct = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)         #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

            print(f"\nStarting Training on resnet50 number {i+1} on {subset_names[i]} band")
            train_losses = model_training(model, dataloader_train_list[i], loss_funct, optimizer, epochs)
            ensemble_losses.append(train_losses)

            model_parameters_name = f'{subset_names[i]}.pth'
            save_model_parameters(model, model_parameters_path, model_parameters_name)
            model_list.append(model) # Net parameters are saved
        

        # VALIDATION
        # We use the validation set to determine the weights used for the ensembling the resnet's (the weight w_i is the accuracy of the resnet i)
        print("\nValidation: start") 
        ensemble_weights = []      
        with tqdm(total=len(model_list), unit='models') as valbar:
            for i, model in enumerate(model_list) :
                model.eval()
                results = model_test(model, dataloader_val_list[i]) 
                ensemble_weights.append(results[0])  # Taking only the accuracy from the method resnet50_test
                valbar.update(1)
        


        # TESTING
        print("\nTesting: start") 
        correct_pred = 0
        predicted_labels = []
        all_classifiers_predictions = []

        with tqdm(total=len(model_list), unit='models') as testbar:
            for i, model in enumerate(model_list) :     # First testing all models and transforming the class predictions in one hot encoding
                model.eval()
                accuracy, predictions, true_labels = model_test(model, dataloader_test_list[i])
                predictions = to_one_hot(predictions, num_classes)
                all_classifiers_predictions.append(predictions)     # list of numpy arrays
                testbar.update(1)

        num_images = len(all_classifiers_predictions[0])
        with tqdm(total=num_images, unit='img') as bar:
            for i in range(num_images) : # For each image

                one_hot_weighted = []
                for j in range(len(model_list)) :   # For each classifier
                    one_hot_weighted.append(all_classifiers_predictions[j][i] * ensemble_weights[j])
                
                one_hot_weighted = np.array(one_hot_weighted)
                result = np.sum(one_hot_weighted, axis=0) # Sum element-wise along the columns
                net_prediction = np.argmax(result)

                predicted_labels.append(net_prediction)
                if net_prediction == true_labels[i] :
                    correct_pred += 1
                bar.update(1)

        # Parameters estimation
        accuracy = correct_pred / num_images
        precision = metrics.precision_score(true_labels, predicted_labels, average='weighted')
        recall = metrics.recall_score(true_labels, predicted_labels, average='weighted')
        f1 = metrics.f1_score(true_labels, predicted_labels, average='weighted')
        plots.show_confusion_matrix(true_labels, predicted_labels)
        plots.plot_train_loss(ensemble_losses, epochs, subset_names)

        print(F'Accuracy = {accuracy}')
        print(f'Precision = {precision}')
        print(f'Recall = {recall}')
        print(f'F1 Score = {f1}')
        

    elif(ANALYSIS == 2) : # PCA ANALYSIS
        print("Starting analysis: PCA layer is added before resnet50 for channel reduction from 13 to 3")
        if not SKIP_PCA : 
            # Normalization 
            mean, std = calculate_mean_std(train_instances, test_instances, bands = 13) 
            train_instances = (train_instances - np.array(mean)) / np.array(std)
            test_instances = (test_instances - np.array(mean)) / np.array(std)

            # Flattening the data in order to apply PCA, (num_img, 64, 64, 13) ---> (num_img, 4096, 13)
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
            dataset.save_to_hdf5(reconstructed_train_imgs, reconstructed_test_imgs, directory_name="PCA_dataset", file_name="imagesPCA.h5")
            print('Done')
        else : # PCA computation already done
            print('Loading modified PCA dataset...')
            reconstructed_train_imgs, reconstructed_test_imgs = dataset.load_hdf5_PCA(file_path="../PCA_dataset/imagesPCA.h5")
            print("Done")

        
        mean, std = calculate_mean_std(reconstructed_train_imgs, reconstructed_test_imgs, bands = 3) 
        transform = transforms.Compose([                            
            transforms.ToPILImage(), 
            transforms.Resize(256),                                                                                          
            transforms.CenterCrop(224),                             
            transforms.ToTensor(),                                                                                                                                                                                                            
            transforms.Normalize(mean = mean, std = std)            
        ])

        # Start computation, on the channel reduced dataset through PCA
        train_dataset_PCA = dataset.EuroSATDataset(reconstructed_train_imgs, train_label, transform)
        test_dataset_PCA = dataset.EuroSATDataset(reconstructed_test_imgs, test_label, transform) 
        
        train_data_loader = torch.utils.data.DataLoader(train_dataset_PCA, batch_size = batch_size, shuffle = True, generator = g_device)
        test_data_loader = torch.utils.data.DataLoader(test_dataset_PCA, batch_size = batch_size, shuffle = True, generator = g_device)


        print("\nTraining: start")    
        # DenseNet structure
        model = densenet161(DenseNet161_Weights.DEFAULT)
        num_features = model.classifier.in_features     # number of features in input in the last FC layer
        model.classifier = torch.nn.Linear(num_features, num_classes)

        # ResNet structure
        #model = resnet50(weights = ResNet50_Weights.DEFAULT)    
        #num_features = model.fc.in_features     # number of features in input in the last FC layer
        #model.fc = torch.nn.Linear(num_features, num_classes)

        model.to(DEVICE)
        loss_funct = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  
        train_losses = model_training(model, train_data_loader, loss_funct, optimizer, epochs)
        
        
        print("\nTesting: start")
        model.eval()
        accuracy, predictions, true_labels = model_test(model, test_data_loader)
        precision = metrics.precision_score(true_labels, predictions, average='weighted')
        recall = metrics.recall_score(true_labels, predictions, average='weighted')
        f1 = metrics.f1_score(true_labels, predictions, average='weighted')
        plots.show_confusion_matrix(true_labels, predictions)
        plots.plot_train_loss(train_losses, epochs)
        
        print(F'Accuracy = {accuracy}')
        print(f'Precision = {precision}')
        print(f'Recall = {recall}')
        print(f'F1 Score = {f1}')
        print("\nTesting: done") 

    else : # DEFAULT, RGB ANALYSIS
        rgb_band = [3,2,1]
        transform = transforms.Compose([                            
            transforms.ToPILImage(), 
            transforms.Resize(256),                                                                                          
            transforms.CenterCrop(224),                             
            transforms.ToTensor(),          
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])                                                                                                                                                                                                                
        ])

        print("STARTING ANALYSIS: resnet50 architecture applied to RGB band")
        print("Creating dataset subsets")
        train_dataset_RGB = dataset.EuroSATDataset(train_instances[:, :, :, rgb_band], train_label, transform)
        test_dataset_RGB = dataset.EuroSATDataset(test_instances[:, :, :, rgb_band], test_label, transform) 
        train_data_loader = torch.utils.data.DataLoader(train_dataset_RGB, batch_size = batch_size, shuffle = True, generator = g_device)
        test_data_loader = torch.utils.data.DataLoader(test_dataset_RGB, batch_size = batch_size, shuffle = True, generator = g_device)
        

        # TRAINING
        # DenseNet structure
        #model = densenet161(DenseNet161_Weights.DEFAULT)
        #num_features = model.classifier.in_features     # number of features in input in the last FC layer
        #model.classifier = torch.nn.Linear(num_features, num_classes)

        # ResNet structure
        model = resnet50(weights = ResNet50_Weights.DEFAULT)
        num_features = model.fc.in_features     # number of features in input in the last FC layer
        model.fc = torch.nn.Linear(num_features, num_classes)
        model.to(DEVICE)
        loss_funct = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)        

        print("\nTraining: start")
        train_losses = model_training(model, train_data_loader, loss_funct, optimizer, epochs)
        
        
        # TESTING
        print("\nTesting: start") 
        model.eval()
        accuracy, predictions, true_labels = model_test(model, test_data_loader)

        precision = metrics.precision_score(true_labels, predictions, average='weighted')
        recall = metrics.recall_score(true_labels, predictions, average='weighted')
        f1 = metrics.f1_score(true_labels, predictions, average='weighted')
        plots.show_confusion_matrix(true_labels, predictions)
        plots.plot_train_loss(train_losses, epochs, subset_names)
        print(F'Accuracy = {accuracy}')
        print(f'Precision = {precision}')
        print(f'Recall = {recall}')
        print(f'F1 Score = {f1}')
    
if __name__ == '__main__':
    main()