import h5py
import torch 
import numpy as np
import scipy.io as sio

def normalize(x):
    """
    Perform Min-Max normalization to scale input data to [0, 1] range

    Parameters
    ----------
    x : numpy.ndarray
        Input data array of any shape containing numerical values

    Returns
    -------
    numpy.ndarray
        Normalized array with same shape as input, where:
        - Minimum value becomes 0
        - Maximum value becomes 1
        - Other values linearly scaled between them

    Notes
    -----
    - Returns all zeros if input has constant values (max == min)
    - Sensitive to outliers due to max/min dependence
    - Common alternative: z-score standardization (mean=0, std=1)
    """
    # Compute normalized values
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    
    return x

class UCIDigit():
    def __init__(self, path):
        data = sio.loadmat(path + 'uci-digit.mat')
        self.Y = data['truth'].astype(np.int32).reshape(2000, )
        self.V1 = data['mfeat_fac'].astype(np.float32)
        self.V2 = data['mfeat_fou'].astype(np.float32)
        self.V3 = data['mfeat_kar'].astype(np.float32)
    def __len__(self):
        return 2000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(normalize(x1)), torch.from_numpy(normalize(x2)), torch.from_numpy(normalize(x3))], self.Y[idx], torch.from_numpy(np.array(idx)).long()
    
class BBCSport():
    def __init__(self, path):
        data = h5py.File(path + 'BBCSport.mat')
        self.Y = np.squeeze(np.array(data['Y'])).astype(np.int32)
        self.V1 = np.array(data[data['X'][0][0]]).T.astype(np.float32)
        self.V2 = np.array(data[data['X'][0][1]]).T.astype(np.float32)
    def __len__(self):
        return 544
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class BBC4view():
    def __init__(self, path):
        data = sio.loadmat(path + 'BBC4view_685.mat')
        self.Y = data['Y'].astype(np.int32).reshape(685, )
        self.V1 = data['X'][0][0].A.astype(np.float32)
        self.V2 = data['X'][0][1].A.astype(np.float32)
        self.V3 = data['X'][0][2].A.astype(np.float32)
        self.V4 = data['X'][0][3].A.astype(np.float32)
    def __len__(self):
        return 685
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class BDGP():
    def __init__(self, path):
        data = sio.loadmat(path + 'BDGP.mat')
        self.Y = data['Y'].T.astype(np.int32).reshape(2500,)
        self.V1 = data['X1'].astype(np.float32)
        self.V2 = data['X2'].astype(np.float32)
    def __len__(self):
        return 2500
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
   

class HW2sources():
    def __init__(self, path):
        data = sio.loadmat(path + 'HW2sources.mat')
        self.Y = data['Y'].astype(np.int32).reshape(2000, )
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
    def __len__(self):
        return 2000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Handwritten():
    def __init__(self, path):
        data = sio.loadmat(path + 'handwritten.mat')
        self.Y = data['Y'].astype(np.int32).reshape(2000, )
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
        self.V4 = data['X'][3][0].astype(np.float32)
        self.V5 = data['X'][4][0].astype(np.float32)
        self.V6 = data['X'][5][0].astype(np.float32)
    def __len__(self):
        return 2000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]
        x6 = self.V6[idx]
        return [torch.from_numpy(normalize(x1)), torch.from_numpy(normalize(x2)), torch.from_numpy(normalize(x3)),
                torch.from_numpy(normalize(x4)), torch.from_numpy(normalize(x5)), torch.from_numpy(normalize(x6))], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class LandUse21():
    def __init__(self, path):
        data = sio.loadmat(path + 'LandUse-21.mat')
        self.Y = np.squeeze(data['Y']).astype(np.int32) 
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][0][1].astype(np.float32)
        self.V3 = data['X'][0][2].astype(np.float32)
    def __len__(self):
        return 2100
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Mfeat():
    def __init__(self, path):
        data = sio.loadmat(path + 'Mfeat.mat')
        self.Y = data['Y'].astype(np.int32).reshape(2000, )
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
        self.V4 = data['X'][3][0].astype(np.float32)
        self.V5 = data['X'][4][0].astype(np.float32)
        self.V6 = data['X'][5][0].astype(np.float32)
    def __len__(self):
        return 2000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]
        x6 = self.V6[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4), torch.from_numpy(x5), torch.from_numpy(x6)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Webkb():
    def __init__(self, path):
        data = sio.loadmat(path + 'webkb.mat')
        self.Y = data['Y'].astype(np.int32).reshape(203, )
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][0][1].astype(np.float32)
        self.V3 = data['X'][0][2].astype(np.float32)
    def __len__(self):
        return 203
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class STL10():
    def __init__(self, path):
        data = h5py.File(path + 'stl10_fea.mat', 'r')
        self.Y = np.squeeze(np.array(data['Y'])).astype(np.int32)
        self.V1 = np.array(np.transpose(data[data['X'][0][0]])).astype(np.float32)
        self.V2 = np.array(np.transpose(data[data['X'][1][0]])).astype(np.float32)
        self.V3 = np.array(np.transpose(data[data['X'][2][0]])).astype(np.float32)
    def __len__(self):
        return 13000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Cifar10():
    def __init__(self, path):
        data = sio.loadmat(path + 'cifar10.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(50000, )
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][1][0].T.astype(np.float32)
        self.V3 = data['data'][2][0].T.astype(np.float32)
    def __len__(self):
        return 50000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Cifar100():
    def __init__(self, path):
        data = sio.loadmat(path + 'cifar100.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(50000, )
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][1][0].T.astype(np.float32)
        self.V3 = data['data'][2][0].T.astype(np.float32)
    def __len__(self):
        return 50000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

def load_data(dataset):
    """
    Load configuration parameters for specified multi-view dataset

    Parameters
    ----------
    dataset : str
        Name of the dataset to load. Supported options:
        - 'uci_digit': UCI Handwritten Digit dataset
        - 'BBCSport': BBC Sports articles
        - 'BBC4view': BBC 4-view dataset
        - 'BDGP': Berkeley Drosophila Genome Project
        - 'HW2sources': Handwritten 2-sources characters
        - 'Handwritten': Multi-feature handwritten digits
        - 'LandUse21': 21-class land use imagery
        - 'Mfeat': Multiple Features dataset
        - 'Webkb': Web page classification
        - 'STL10': 10-class image recognition
        - 'Cifar10/100': CIFAR image datasets

    Returns
    -------
    tuple
        dataset: Initialized dataset object
        dims: List[int] - Feature dimensions per view
        view: int - Number of data views/modalities
        data_size: int - Total number of samples
        class_num: int - Number of target classes

    Raises
    ------
    NotImplementedError
        For unsupported dataset names

    Notes
    -----
    - All datasets are loaded from '../../datasets/' relative path
    - Each dataset class should implement proper data loading interface
    """
    path = './data'
    if dataset == 'uci_digit':
        dataset = UCIDigit(path=path)
        dims = [216, 76, 64]
        view = 3
        data_size = 2000
        class_num = 10

    elif dataset == 'BBCSport':
        dataset = BBCSport(path=path)
        dims = [3183, 3203]
        view = 2
        data_size = 544
        class_num = 5

    elif dataset == 'BBC4view':
        dataset = BBC4view(path=path)
        dims = [4659, 4633, 4665, 4684]
        view = 4
        data_size = 685
        class_num = 5

    elif dataset == 'BDGP':
        dataset = BDGP(path=path)
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
        
    elif dataset == 'HW2sources':
        dataset = HW2sources(path=path)
        dims = [784, 256]
        view = 2
        data_size = 2000
        class_num = 10

    elif dataset == 'Handwritten':
        dataset = Handwritten(path=path)
        dims = [216, 76, 64, 6, 240, 47]
        view = 6
        data_size = 2000
        class_num = 10

    elif dataset == 'LandUse21':
        dataset = LandUse21(path=path)
        dims = [20, 59, 40]
        view = 3
        data_size = 2100
        class_num = 21

    elif dataset == 'Mfeat':
        dataset = Mfeat(path=path)
        dims = [216, 76, 64, 6, 240, 47]
        view = 6
        data_size = 2000
        class_num = 10
    
    elif dataset == 'Webkb':
        dataset = Webkb(path=path)
        dims = [1703, 230, 230]
        view = 3
        data_size = 203
        class_num = 4

    elif dataset == 'STL10':
        dataset = STL10(path=path)
        dims = [1024, 512, 2048]
        view = 3
        data_size = 13000
        class_num = 10

    elif dataset == 'Cifar10':
        dataset = Cifar10(path=path)
        dims = [512, 2048, 1024]
        view = 3
        data_size = 50000
        class_num = 10

    elif dataset == 'Cifar100':
        dataset = Cifar100(path=path)
        dims = [512, 2048, 1024]
        view = 3
        data_size = 50000
        class_num = 100

    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num