#from models.detector import Detector
#from models.classifier import Classifier
#from models.corrector import Corrector
import pandas as pd
import numpy as np


#detector = Detector()
#classifier = Classifier()
#corrector = Corrector()


dataset = pd.read_csv('data/dataset_cleand-1-1-1.csv')

dataset=dataset[['native', 'text']]

for i in dataset.index:
    if '<br/>' in dataset['native'][i]:
        dataset['native'][i]=dataset['native'][i].split('<br/>')
    else:
        dataset['native'][i]=np.array([dataset['native'][i]])

print(dataset.head())