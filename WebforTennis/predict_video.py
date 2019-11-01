# Dependencies
import pandas as pd
import os

# Dependencies
from bs4 import BeautifulSoup
import requests
import pymongo

def predict(modelName, fileName):
    filePath = "C:/bootcamp/Final/TennisClassCountsPro/videoClassification-master/"
    # commandString = 'python ' + filePath + 'predict_video.py -m ' +  filePath + 'model_non_pro -l '+ filePath + 'model_non_pro/lb.pickle -i '  + filePath + 'example_clips/IMG_0527.MOV -s 30'
    commandString = 'python ' + filePath + 'predict_video.py -m ' +  filePath + modelName +' -l '+ filePath + modelName + '/lb.pickle -i '  + filePath + 'example_clips/' + fileName +' -s 30'
    print("command = ", commandString)
    os.system(commandString)
    return 'success'