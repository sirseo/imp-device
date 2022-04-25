import gc
import time
import random
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

#############################
#           ARG             #
#############################

ECG_MIN = 300
ECG_MAX = 600
DATA_COUNT = 10000

EPOCH = 5000
BATCH_SIZE = 256

#############################
#      UNKNOWEN FUNC        #
#############################

def randomECGValue(ECG_MIN, ECG_MAX):
    return random.random() * (ECG_MAX- ECG_MIN) + ECG_MIN

def createECGSeq(seqSize = 8):
    seq = []
    for j in range(seqSize):
        seq.append(randomECGValue(ECG_MIN, ECG_MAX))
    return seq

randECGSeqDf = pd.DataFrame(createECGSeq(seqSize = DATA_COUNT))
MEAN = randECGSeqDf.mean()
STD = randECGSeqDf.std()


#############################
#           MAIN            #
#############################
def train(inCsvPath, outH5Dir):
    train_data = pd.read_csv(inCsvPath)
    test_data = pd.read_csv('./csv/test.csv')
    #shuffle data
    train_data = train_data.sample(frac=1).reset_index(drop = True)
    test_data = test_data.sample(frac=1).reset_index(drop = True)
    # create x_train and x_test
    x_train = []
    x_train_label = []
    for data_row in train_data.values:
        x_train.append(pd.DataFrame(data_row[0:8]))
        x_train_label.append(pd.DataFrame(data_row[8:]))
    x_train = np.stack(x_train)
    x_test = []
    x_test_label = []
    for data_row in test_data.values:
        x_test.append(pd.DataFrame(data_row[0:8]))
        x_test_label.append(pd.DataFrame(data_row[8:]))
    x_test = np.stack(x_test)
    #prepare Model
    model = keras.Sequential(
        [
            layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
            layers.Conv1D(filters=32, kernel_size=7, padding="same", strides=2, activation="relu"),
            layers.Dropout(rate=0.2),
            
            layers.Conv1D(filters=16, kernel_size=7, padding="same", strides=2, activation="relu"),
            layers.Conv1DTranspose(filters=16, kernel_size=7, padding="same", strides=2, activation="relu"),
            layers.Dropout(rate=0.2),
            
            layers.Conv1DTranspose(filters=32, kernel_size=7, padding="same", strides=2, activation="relu"),
            layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
        ]
    )
    outH5Path = outH5Dir+'/'+str(time.time())+"_model.h5"
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    checkpoint = ModelCheckpoint(outH5Path, 
                                monitor='val_loss', 
                                verbose=1, 
                                save_best_only=True, 
                                mode='auto', 
                                save_weights_only = True)
    earlystopping = EarlyStopping(monitor="val_loss", patience=10)
    history = model.fit(
        x_train,
        x_train,
        epochs=EPOCH,
        batch_size=BATCH_SIZE,
        validation_split=0.3,
        callbacks=[checkpoint, earlystopping],
    )
    model.load_weights(outH5Path)
    x_train_pred = model.predict(x_train)
    train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis = 1)
    threshold = np.max(train_mae_loss)
    print(threshold)
    x_test_pred = model.predict(x_test)
    test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis = 1)
    # find fails
    notAbnormalButDetectedCount = 0
    abnormalButNotDetectedCount = 0
    for i in range(len(x_test)):
        #print(x_test[i], test_mae_loss[i])
        #print(x_test_label[i][0][0])
        if(test_mae_loss[i] > threshold and x_test_label[i][0][0] == 0):
            notAbnormalButDetectedCount += 1
        if(test_mae_loss[i] < threshold and x_test_label[i][0][0] == 1):
            abnormalButNotDetectedCount += 1
    print("{0}({1}, {2}) / {3} | {4}% ".format(
        notAbnormalButDetectedCount+ abnormalButNotDetectedCount,
        notAbnormalButDetectedCount,
        abnormalButNotDetectedCount,
        len(x_test),
        (notAbnormalButDetectedCount+ abnormalButNotDetectedCount)/len(x_test)*100
    ))
    print(threshold)
    del [[train_data, test_data]]
    gc.collect()
    return (threshold, outH5Path)

def trainWithNewValue(inCsvPath, ecgDataAndAbnormal, outCsvDir, outH5Dir):
    prevDf = pd.read_csv(inCsvPath) 
    newDf = pd.DataFrame(
        ecgDataAndAbnormal,
        columns = ['seq1','seq2','seq3','seq4','seq5','seq6','seq7','seq8','abnormal'])
    newDf = newDf.append(prevDf, ignore_index=True) 
    outCsvPath = outCsvDir+'/'+str(time.time())+"_train.csv"
    newDf.to_csv(outCsvPath, index=False)
    del [[newDf, prevDf]]
    gc.collect()
    trainRes = train(outCsvPath,outH5Dir)   
    return outCsvPath,  trainRes[1], trainRes[0]
