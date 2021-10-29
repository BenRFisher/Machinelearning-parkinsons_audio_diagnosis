import numpy as np
import pandas as pd
import sklearn
import os,sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from scipy.io.wavfile import read
import librosa
import parselmouth
from parselmouth.praat import call

#extracts data from .csv file (no file loc-add it yourself)
def datareader():
    data=pd.read_csv('parkinsons.data')
    data=data[['name','MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ','Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA','HNR','status']]
    features=data.loc[:,data.columns!='status'].values[:,1:]
    labels=data.loc[:,'status'].values
    return features, labels
#since the features are of different ranges, apply minmax scaling 
def featurescaler(features):
    minmax=MinMaxScaler((-1,1))
    x=minmax.fit_transform(features)
    return x,minmax
#create a train/test split of the data
def datasplit(x,labels):
    x_train,x_test,y_train,y_test=train_test_split(x,labels,test_size=0.2,random_state=7)
    return x_train,x_test,y_train,y_test
#train a gradient boosting classifier on the data
def modeltraining(x_train,y_train):
    classifier=GradientBoostingClassifier()
    classifier.fit(x_train,y_train)
    return classifier
#test the model to calculate accuracy
def modeltest(classifier, x_test,y_test):
    pred=classifier.predict(x_test)
    acc=accuracy_score(y_test, pred)
    return acc
#download voice clip data, and analyse using parselmouth (praat python extension)
def measurePitch(voiceID, f0min, f0max, unit):
    sound = parselmouth.Sound(voiceID) # read the sound
    duration = call(sound, "Get total duration") # duration
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
    #minF0 = call(pitch,"Get minimum pitch", 0, 0, unit)
    #maxF0 = call(pitch,"Get maximum pitch", 0, 0, unit)
    
    stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    return duration, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer
#this translates binary output prediction into positive/negative parkinsons diagnosis
def diagnose(diagnosis):
    if diagnosis==1:
        print('patient has Parkinsons')
    else:
        print('patient does not have Parkinsons')

def main():
    voiceID='test.wav'
    features,labels = datareader()
    x,minmax=featurescaler(features)
    x_train,x_test,y_train,y_test = datasplit(x,labels)
    classifier=modeltraining(x_train,y_train)
    print(modeltest(classifier, x_test,y_test))
    duration, meanF0, stdevF0, hnr,localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer= measurePitch(voiceID, 75, 500, "Hertz")
    val=[meanF0,500,75,localJitter,localabsoluteJitter,rapJitter,ppq5Jitter,ddpJitter,localShimmer,localdbShimmer,apq3Shimmer,aqpq5Shimmer,apq11Shimmer,ddaShimmer,hnr]    
    #data has to be reshaped to go through minmax scaling and then reshaped back, so 
    #i converted into a numpy array then reshaped it 
    val=np.array(val).reshape(-1,1)
    val=minmax.fit_transform(val)
    val=val.reshape(1,-1)
    diagnosis=classifier.predict(val)
    diagnose(diagnosis)



if __name__ == "__main__":
    main()

