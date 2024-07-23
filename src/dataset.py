import os                           # paths
import h5py                         # compression of large dataset and upload
import tifffile                     # reading and writing TIFF files
import torch
import numpy as np                  # arrays 
from torch.utils.data import Dataset


class EuroSATDataset(Dataset) : 
    """Eurosat dataset object.

    Args:
        instances (list):       list of instances
        labels (list):          list of labels
        transform (callable):   transformations applied to the instances
    """
    def __init__(self, instances, labels, transform):
        self.labels = labels                     # classes
        self.instances = instances               # images
        self.transform = transform               # transformation

    def __len__(self) :
        return len(self.instances)
    
    def __getitem__(self, idx) :
        return self.transform(self.instances[idx]), torch.tensor(self.labels[idx] , dtype=torch.long)
    
    def __getlabels__(self) : 
        return self.labels

    def __getchannelesvalue__(self, bands_selected) :
        return self.transform(self.instances[:,:,bands_selected]), self.labels[:]
    


def create_EuroSATDatasets(instances, labels, subset_bands, transform) :
    """Create different EuroSATDataset objects by selecting from instances the given subset of bands. 

    Args:
        instances (numpy array):    images of size (64, 64, 13) each
        labels (numpy array):       labels
        subset_bands (list):        indexes to be selected  
        transform (callable):       transformations applied to the instances

    Returns:
        A list of EuroSATDataset objects
    """
    datasets = []
    for bands in subset_bands:
        dataset = EuroSATDataset(instances[:, :, :, bands], labels, transform)
        datasets.append(dataset)
    return datasets



def create_data_loaders(train_datasets, val_datasets, test_datasets, batch_size, generator):
    """Create separate torch DataLoaders for each EuroSATDataset object inside the given train/validation/test lists 

    Args:
        train_datasets (list):          list with EuroSATDataset objects
        val_datasets (list):            list with EuroSATDataset objects
        test_datasets (list):           list with EuroSATDataset objects
        batch_size (int):               size of the batches used during training
        generator (torch.Generator):

    Returns:
        Three lists of DataLoaders
    """
    train_loaders = [torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator) for dataset in train_datasets]
    val_loaders = [torch.utils.data.DataLoader(dataset, batch_size=batch_size) for dataset in val_datasets]
    test_loaders = [torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True) for dataset in test_datasets]
    return train_loaders, val_loaders, test_loaders


def load_img(img_path) :
    """Read an image through the tifffile library and convert the values of the image from uint16 to float32. 

    Args:
        img_path (string): path to a .tif image

    Returns:
        The image as numpy array
    """
    img_np_array = tifffile.imread(img_path)
    return img_np_array.astype(np.float32)


def save_to_hdf5(train_instances, test_instances, directory_name, file_name, train_label = None, test_label = None) :
    """Save the given numpy arrays in a .h5 format by using the h5py library. The data size is reduced thanks to a lossless compression.
       In case the specified directory is not found, the directory is created.

    Args:
        train_instances (numpy array):  train instances (num train imgs, 64, 64, 13)
        test_instances (numpy array):   test instances (num test imgs, 64, 64, 13)
        directory_name (string):        name of the directory where the data is saved 
        file_name (string):             name of the file to be saved
        train_label(numpy array):       optional, train labels 
        test_label (numpy array):       optional, test labels
    """
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
    """Load a .h5 file with the PCA version of the EUROSAT dataset previously stored.

    Args:
        file_path (string): path of a .h5 file
    """
    with h5py.File(f'{file_path}', 'r') as hf :
            train_instances = hf['train']['data'][:]
            test_instances = hf['test']['data'][:]

    return train_instances, test_instances


def load_hdf5_EUROSAT(file_path) : 
    """Load a .h5 file with the EUROSAT dataset (13 bands) previously stored.

    Args:
        file_path (string): path of a .h5 file
    """
    with h5py.File(f'{file_path}', 'r') as hf :
            train_instances = hf['train']['data'][:]
            train_labels = hf['train']['labels'][:]
            test_instances = hf['test']['data'][:]
            test_labels = hf['test']['labels'][:]

    return train_instances, train_labels, test_instances, test_labels
