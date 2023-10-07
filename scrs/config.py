""""
Dataset configurations:
  :param TRAIN_FILEPATH -> the directory path to dataset files of training
  :param VALID_FILEPATH -> the directory path to dataset files of validation
  :param SAVE_PATH -> the directory path to save plot and the generator model
""""

TRAIN_FILEPATH = "./TRAIN_files/"      # Example: Train_1.npz, Train_2.npz, Train_3.npz ... (src_images1, src_images2, src_images3, src_images4, src_images5, tar_images)
VALID_FILEPATH = "./VAILD_files/Validation.npz"      # Valid.npz (src_images1, src_images2, src_images3, src_images4, src_images5, tar_images)
SAVE_PATH = "./SAVE_CE-MRI/"

""""
Training configurations:
  :param SEED -> the random seed
  :param IMAGE_SHAPE -> image shape
  :param LOSSES_WEIGHTS -> the weights for losses
  :param TRAINING_EPOCH -> number of training epochs
  :param NUNBER_BATCH  -> specifies the batch size of training process 
""""

SEED = 1203
IMAGE_SHAPE = (224, 224, 1)  
LOSSES_WEIGHTS = [1,5,5,5,5,5,100]
TRAINING_EPOCH = 100
NUMBER_BATCH = 8
