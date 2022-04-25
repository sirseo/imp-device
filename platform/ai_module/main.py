import json
from flask import Flask, render_template, request
import mobiusAPI as ml
import predict as ep
import train as et
import time
import logging

serverURL="http://mobius:7579"
thisURL = "http://ai_module"

TH = 1
ECG_SEQ_DATA_PATH = './data/train/train1.csv'
MODEL_PATH = './model/model.pt'
isTraining = False
dataToTrain = []

def subAdcAI(serverURL, ae_name, subURL, role):
    AEname = ae_name
    INCSEurl = serverURL
    ml.createSubscription(AEname, "/deviceHealthData/adcData/rawValue/AI", subURL, INCSEurl, role)

def subTrainAI(serverURL, ae_name, subURL, role):
    AEname = ae_name
    INCSEurl = serverURL
    ml.createSubscription(AEname, "/deviceHealthData/adcData/train", subURL, INCSEurl, role)

app = Flask(__name__)

@app.route('/')
def aliveCheck():
    print("alive check")
    return json.dumps({"message":"working"})

@app.route('/cse', methods = ['POST'])
def cseCreated():
    time.sleep(3)
    cse = request.get_json()
    try:
        ae_name = cse['m2m:sgn']['nev']['rep']['m2m:ae']['rn']
        subAdcAI(serverURL, ae_name, thisURL+"/adc_ai", "manager")
        subTrainAI(serverURL, ae_name, thisURL+"/train", "manager")
    except KeyError:
        logging.info("not ae\n")
    return json.dumps({"message":"working"})


@app.route('/adc_ai', methods = ['POST'])
def aiDataUpdated():
    data = request.get_json()
    try:
        data = data['m2m:sgn']['nev']['rep']['m2m:cin']
        ecgData = data['con']
        userName = data['cr']
        ecgDataIndex = ecgData['index']
        rawDataList = ecgData['rawdatalist']
        rawDataList = json.loads(rawDataList)
        predictValue = ep.predict(MODEL_PATH, rawDataList)
        #print(rawDataList,predictValue)
        result = 0
        if( predictValue > TH):
            result = 1
        jsonrawdata = {
        "index": str(ecgDataIndex),
        "result": str(result)
        }
        ml.createContentInstance(userName, "/deviceHealthData/result", jsonrawdata, serverURL, "manager")
    except KeyError:
        logging.info("subscription init\n")

    return json.dumps({"message":"working"})


@app.route('/train', methods = ['POST'])
def trainDataUpdated():
    data = request.get_json()
    try:
        data = data['m2m:sgn']['nev']['rep']['m2m:cin']
        trainData = data['con']
        userName = data['cr']
        rawDataList = trainData['rawdatalist']
        rawDataList = json.loads(rawDataList)
        abnormal = trainData['abnormal']
        abnormal = int(abnormal)
        global isTraining
        global dataToTrain
        if(isTraining):
            dataToTrain.append(rawDataList+[abnormal])
        else:
            isTraining = True
            trainData = dataToTrain
            dataToTrain = []
            global TH
            global ECG_SEQ_DATA_PATH
            global MODEL_PATH
            res = et.trainWithNewValue(ECG_SEQ_DATA_PATH, trainData, './csv/', './model/')
            TH = res[2]
            ECG_SEQ_DATA_PATH = res[0]
            MODEL_PATH = res[1]
            isTraining = False
    except KeyError:
        logging.info("subscription init\n")
         
    return json.dumps({"message":"working"})

#########################################
#                                       #
#           main start                  #
#                                       #
#########################################


#initial code, get AE list and create subscription foreach aes
ml.createSubscription("", "", thisURL+"/cse", serverURL, "manager")
AEs = ml.getOnlyIDUserAEs(serverURL, "manager")
for ae in AEs: 
    subAdcAI(serverURL, ae, thisURL+"/adc_ai", "manager")
    subTrainAI(serverURL, ae, thisURL+"/train", "manager")
#sub for training_data
app.run(host='0.0.0.0', port=80, debug=True)

