import os.path
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

# Gets home dir cross platform
# "D:\pycharmProject"
HOME = BASE_DIR
MyName = "SimpleTransferLearning-Pytorch-master"

# Path to store checkpoint model
CheckPoints = 'checkpoints'
CheckPoints = os.path.join(HOME, MyName, CheckPoints)

# Results
Results = 'results'
Results = os.path.join(HOME, MyName, Results)

# Path to store tensorboard load
tensorboard_log = 'tensorboard'
tensorboard_log = os.path.join(HOME, MyName, tensorboard_log)

# Path to save log
log = 'log'
log = os.path.join(HOME, MyName, log)

# Path to save classification train log
classification_train_log = 'classification_train'

# Data
DATAPATH = os.path.dirname(BASE_DIR)

# CIFAR10
CIFAR_path = os.path.join(DATAPATH, 'data', 'cifar')

# mydataset pets
PETS_path = os.path.join(DATAPATH, 'data', 'pet')
