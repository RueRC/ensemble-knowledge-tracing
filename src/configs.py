from os.path import abspath, dirname
from pathlib import Path

DATASET_PATH = Path(abspath(Path(dirname(__file__)) / '..' / 'Dataset'))
ASSIST2017_PATH = Path(DATASET_PATH / 'assist2017.csv') # original public dataset
ASSIST2017_SEQ_PATH = Path(DATASET_PATH / 'assist2017_seq.csv')
ASSIST2017_SEQ_TRAIN_PATH = Path(DATASET_PATH / 'assist2017_seq_train.csv')
ASSIST2017_SEQ_TEST_PATH = Path(DATASET_PATH / 'assist2017_seq_test.csv')
ASSIST2017_FEA_PATH = Path(DATASET_PATH / 'assist2017_fea.csv')