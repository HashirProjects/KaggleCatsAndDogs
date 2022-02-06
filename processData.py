import os
import cv2
import random
import numpy as np
import pickle

#i used these functions to process the raw images. The result of this process is saved in the text files trainingValues and trainingResults

resolution = 50
def loadData(noOfImgs):
    
    dataSet=[]
    
    for category in os.listdir(DIR):
        
        categoryDir=os.path.join(DIR,category)

        counter=0
        
        for img in os.listdir(categoryDir):
            if counter < noOfImgs:
                
                imgArrayForm = cv2.imread(os.path.join(categoryDir,img),0)
                counter= counter+1
            
                try:
                    imgArrayForm= cv2.resize(imgArrayForm, (resolution,resolution))#lowers resolution to increase training speed and reduce memory problems
                    dataSet.append([imgArrayForm,category])
                except:
                    print(img)#just so I can see which images are broken
            else:
                break
            
    with open("ImgData.txt", "wb") as file:
        pickle.dump(dataSet, file)
                
    return dataSet

def createDataSet():

    DataSet=[]
    
    with open("ImgData.txt", "rb") as file:
        DataSet= pickle.load(file)
        
    random.shuffle(DataSet)
    
    Values=[]
    results=[]
    
    for x in DataSet:
        
        Values.append(x[0])
        if x[1] == 'Cat':
            results.append([0,1])
        else:
            results.append([1,0])
    
    trainingResults = np.array(results)
    trainingValues = np.array(Values)

    trainingValues= trainingValues/255#makes all pixel values between 0 and 1 (helps to increase training speed if training values are closer together)
    
    with open("trainingValues.txt", "wb") as file:
        pickle.dump(trainingValues, file)
    
    with open("trainingResults.txt", "wb") as file:
        pickle.dump(trainingResults, file)