#from models.detector import Detector
#from models.classifier import Classifier
#from models.corrector import Corrector
import pandas as pd
import numpy as np
import language_tool_python
import warnings



def is_english(s, max_ratio = 0.6):
    s = str(s).strip()
    if not s:
        return False

    matches = tool.check(s)
    words = s.split()
    
    if not words:
        return False

    errors = sum(1 for m in matches if "MORFOLOGIK_RULE_EN_GB" in m.ruleId)

    ratio = errors / len(words)

    return ratio <= max_ratio


def dataset_cleaner(dataset):
    print(dataset.shape)
    dataset=dataset[['native', 'text']]

    dataset = dataset.dropna()
    dataset = dataset.drop_duplicates()

    print(dataset.shape)        

    for i in dataset.index:
        if not is_english(dataset['text'][i]):
            #print("Dropped non-english sentence: ", dataset['text'][i])
            dataset = dataset.drop(i)
        else:
            if '<br/>' in dataset['native'][i]:
                dataset['native'][i]=dataset['native'][i].split('<br/>')
            else:
                dataset['native'][i]=np.array([dataset['native'][i]])
    
    print(dataset.shape)
    print(dataset.head())


#detector = Detector()
#classifier = Classifier()
#corrector = Corrector()

warnings.filterwarnings("ignore")
dataset = pd.read_csv('data/dataset_cleand-1-1-1.csv')
tool = language_tool_python.LanguageTool('en-GB')

dataset_cleaner(dataset)

dataset.to_csv('data/dataset_cleaned.csv', index=False)

