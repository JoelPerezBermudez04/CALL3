import language_tool_python
import pandas as pd
import numpy as np
import os
tool = language_tool_python.LanguageTool('en-GB')
dataset = pd.read_csv('./data/dataset_cleaned.csv')
def dataset_detector(dataset):
    ruleId = []
    replacements = []
    category = []
    error_text=[]

    for i in dataset.index:
        text = dataset.loc[i, 'text']
        matches = tool.check(text)
        error_text.append(";".join([ text[m.offset : m.offset + m.errorLength] for m in matches]))
        ruleId.append("; ".join([m.ruleId for m in matches]))
        replacements.append([m.replacements for m in matches])
        category.append("; ".join([m.category for m in matches]))
        if i%1000==0:
            print(f"Processed {i}/{len(dataset)}")

    dataset['error_text']= error_text
    dataset['ruleId'] = ruleId
    dataset['replacements'] = replacements
    dataset['category'] = category
    return dataset

detected_dataset=dataset_detector(dataset)
base_dir = os.path.dirname(os.path.dirname(__file__)) 
save_path = os.path.join(base_dir, "data", "dataset_detected.csv")

detected_dataset.to_csv(save_path, index=False)

# ruleId-----0
# message
# replacement------2
# offsetInContext
# context
# Offset
# errorLength
# category----7
# ruleIssueType
# sentence