#from models.detector import Detector
#from models.classifier import Classifier
#from models.corrector import Corrector
import pandas as pd
import numpy as np
from langdetect import detect
import language_tool_python
import warnings



def is_english1(s):

    try:
        return detect(str(s)) == "en"
    except:
        return False

def is_english2(s, max_ratio = 0.6):
    s = str(s).strip()
    if not s:
        return False

    matches = tool.check(s)
    words = s.split()
    
    # Si no hay palabras, descartamos
    if not words:
        return False

    # Contamos errores que son ortográficos o de palabra desconocida
    errores = sum(1 for m in matches if "MORFOLOGIK_RULE_EN_GB" in m.ruleId)
    
    # Ratio de errores vs palabras
    ratio = errores / len(words)

    return ratio <= max_ratio



#detector = Detector()
#classifier = Classifier()
#corrector = Corrector()

warnings.filterwarnings("ignore")
dataset = pd.read_csv('data/dataset_cleand-1-1-1.csv')
tool = language_tool_python.LanguageTool('en-GB')

print(dataset.shape)
dataset=dataset[['native', 'text']]

dataset = dataset.dropna()
dataset = dataset.drop_duplicates()

print(dataset.shape)

for i in dataset.index:
    if not is_english2(dataset['text'][i]):
        print("Dropped non-english sentence: ", dataset['text'][i])
        dataset = dataset.drop(i)

print(dataset.shape)

for i in dataset.index:
    if '<br/>' in dataset['native'][i]:
        dataset['native'][i]=dataset['native'][i].split('<br/>')
    else:
        dataset['native'][i]=np.array([dataset['native'][i]])

print(dataset.head())

