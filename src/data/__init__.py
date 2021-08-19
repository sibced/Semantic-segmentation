import os.path

# CONSTANTS: navigate in the project directory
DATA_FOLDER = os.path.dirname(os.path.realpath(__file__))
TRAINING_SET_FOLDER_OLD = os.path.join(DATA_FOLDER, 'semantic_drone_dataset', 'training_set')
INPUT_IMAGES_FOLDER_OLD = os.path.join(TRAINING_SET_FOLDER_OLD, 'images')
LABEL_IMAGES_FOLDER_OLD = os.path.join(TRAINING_SET_FOLDER_OLD, 'gt', 'semantic', 'label_images')

TRAINING_SET_FOLDER = os.path.join(DATA_FOLDER, 'semantic_drone_dataset', 'semantic_drone_dataset')
INPUT_IMAGES_FOLDER = os.path.join(TRAINING_SET_FOLDER, 'original_images')
LABEL_IMAGES_FOLDER = os.path.join(TRAINING_SET_FOLDER, 'label_images_semantic')

TRAIN_CSV = os.path.join(DATA_FOLDER, 'train.csv')
TEST_CSV = os.path.join(DATA_FOLDER, 'test.csv')
VALIDATION_CSV = os.path.join(DATA_FOLDER, 'validation.csv')