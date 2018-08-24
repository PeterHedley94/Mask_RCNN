from utils.common import *
import keras
from utils.folder_manipulation import *
from scipy import stats

#CREATE FOLDER SUBSTRUCTURE REQUIRED TO RUN MODEL
def create_folder_structure():
    folders = ["MODEL_OUTPUTS",os.path.join("MODEL_OUTPUTS","logs"),os.path.join("MODEL_OUTPUTS","checkpoints")]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


#AFTER TRAINING MOVE ALL MODEL OUTPUTS TO RELEVANT FOLDERS WITH CORRECT NAMING
def clean_up(model_name):
    to_clean = [TENSORBOARD_LOGS_FOLDER,JSON_LOG_FILE]#,MODEL_SAVE_FOLDER]
    to_make = [TENSORBOARD_OLD_LOGS_FOLDER,JSON_OLD_LOGS_FOLDER]#,OLD_MODELS_FOLDER]
    extension = ['','.json']#,'.hdf5']

    for ext,folder,to_move in zip(extension,to_make,to_clean):
        if not os.path.exists(folder):
            os.makedirs(folder)
        old_list = os.listdir(folder)
        numbers = []
        for i in old_list:
            numbers.append(int(i.split('_')[0]))
        numbers = sorted(numbers)
        if len(numbers) > 0:
            count = numbers[-1] +1
        else:
            count = 0
        new_name = str(count) + '_' + model_name + ext
        os.rename(to_move, os.path.join(folder ,new_name))
        print("Data is in : " + to_move + "/" + new_name)


#
def find_incorrect_classifications(directory_, NN): # pragma: no cover
    incpred = "incorrect_predictions"
    if not os.path.exists(incpred):
        os.makedirs(incpred)

    # Test 1 check if untrained model returns uniform predictions
    folders = get_folders(directory_)
    category = 0
    count = 0
    incorrect = 0
    preds = []
    import shutil
    for folder in folders:

        if not os.path.exists(os.path.join(incpred, folder)):
            os.makedirs(os.path.join(incpred, folder))

        image_list = get_image_names(os.path.join(directory_, folder))
        for image in image_list:
            filepath = os.path.join(directory_, folder, image)
            resized_image = get_image(filepath)#np.squeeze(get_image(filepath),axis=0)

            predictions = NN.predict(resized_image)
            if not os.path.exists(os.path.join(incpred, str(np.argmax(predictions)))):
                os.makedirs(os.path.join(incpred, str(np.argmax(predictions))))
            fileto2 = os.path.join(incpred, str(np.argmax(predictions)), str(count) + image)
            shutil.copyfile(filepath, fileto2)
            preds.append(np.argmax(predictions))
            if np.argmax(predictions) != category:
                fileto = os.path.join(incpred, folder, image)
                shutil.copyfile(filepath, fileto)
                incorrect += 1
            count += 1

        category = category + 1
        print("***********************************************")
        saved = stats.mode(np.array(preds)).mode[0]
        preds = np.array(preds)
        print(saved)
        print(preds)
        print("Accuracy is : " + str(preds[preds==saved].shape[0]/preds.shape[0]) + " on class " + str(category))
        incorrect = 0
        count = 0
        preds = []

#RETURNS A CALLBACK OBJECT WITH THE RELEVANT PARAMETERS FOR TENSORBOARD
def get_Tensorboard(): # pragma: no cover
    tensor_log = keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOGS_FOLDER,
                                histogram_freq=0,
                                batch_size=BATCH_SIZE,
                                write_graph=True,
                                write_grads=False,
                                write_images=True,
                                embeddings_freq=0,
                                embeddings_layer_names=None,
                                embeddings_metadata=None)
    return tensor_log

#RETURNS A CALLBACK OBJECT TO SAVE THE MODEL AFTER EACH EPOCH
def get_model_checkpoint(): # pragma: no cover
    model_check = keras.callbacks.ModelCheckpoint(filepath=INTERMEDIATE_FILE,
                                    monitor='val_loss',
                                    verbose=0,
                                    save_best_only=False,
                                    save_weights_only=False,
                                    mode='auto', period=1)
    return model_check
