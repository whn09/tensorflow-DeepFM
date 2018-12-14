
# set the path-to-files
# TRAIN_FILE = "../../dataset/log_preprocess/20181128/train_ph.libsvm"
# TEST_FILE = "../../dataset/log_preprocess/20181128/test_ph.libsvm"
# TRAIN_FILE = "../../dataset/log_preprocess/20181128/train_ph.fm"
TRAIN_FILE = "../../dataset/log_preprocess/20181128/test_ph.fm"
TEST_FILE = "../../dataset/log_preprocess/20181128/test_ph.fm"

SUB_DIR = "./output"


NUM_SPLITS = 3
RANDOM_SEED = 2018

# types of columns of the dataset dataframe
CATEGORICAL_COLS = []

NUMERIC_COLS = [str(i) for i in range(899)]

IGNORE_COLS = []
