WORK_DIR = "workdir"
REPOSITORY_NAME = ""
GIT_LINK = ""
DATA_SET = ["train", "aug_dataset"]
# dataset 以外で送信したいファイルがあれば記載する
RSYNC_FILES = ["train_master.tsv"]
RSYNC_FILES.extend(DATA_SET)

LOG_NAME = "sample"
BATCH_SIZE = 9
NUM_CLASSES = 1
MAX_EPOCHS = 100
