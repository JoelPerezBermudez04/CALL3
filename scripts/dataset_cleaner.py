import warnings
import language_tool_python
import pandas as pd
import numpy as np
import string

warnings.filterwarnings("ignore")
tool = language_tool_python.LanguageTool('en-GB')
allowed_chars = set(string.ascii_letters + string.digits + string.punctuation + " \n\t")

def is_english(s, max_ratio = 0.6, max_nonlatin = 0.5):
    s = str(s).strip()
    if not s:
        return False

    total_chars = len(s)
    if total_chars == 0:
        return False
    
    nonlatin_chars = 0

    for c in s:
        if c not in allowed_chars:
            nonlatin_chars += 1

    ratio_nonlatin = nonlatin_chars / total_chars

    if ratio_nonlatin > max_nonlatin:
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

        if i % 1000 == 0:
            print(f"Processed {i}/{len(dataset)} rows")


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

    return dataset

dataset = pd.read_csv('../data/dataset_cleand-1-1-1.csv')

cleaned_dataset = dataset_cleaner(dataset)

cleaned_dataset.to_csv('../data/dataset_cleaned.csv', index=False)