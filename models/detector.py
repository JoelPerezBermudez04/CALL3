import language_tool_python
import pandas as pd
import numpy as np
import os
import errant
import spacy

nlp = spacy.load('en_core_web_sm')

annotator = errant.load('en',nlp)

dataset = pd.read_csv('./data/dataset_corrected.csv')

def dataset_detector(input,output):
    
    orig = annotator.parse(input)
    cor = annotator.parse(output)
    alignment = annotator.align(orig, cor)
    edits = annotator.merge(alignment)

    error_text = []
    correct_word=[]
    error_type=[]

    for e in edits:
        e = annotator.classify(e)
        error_text.append(e.o_str)       
        correct_word.append(e.c_str)      
        error_type.append(e.type) 

    return error_text, correct_word, error_type

def dataset_with_errant(dataset):
    all_error_text = []
    all_corrections = []
    all_error_type = []

    for i in dataset.index:
        text = dataset.loc[i, 'text'] 
        corrected = dataset.loc[i, 'corrected_text'] 

        err_text, corr, err_type = dataset_detector(text, corrected)

        all_error_text.append(";".join(err_text))
        all_corrections.append(";".join(corr))
        all_error_type.append(";".join(err_type))

        if i % 500 == 0:
            print(f"Processed {i}/{len(dataset)}")

    dataset['error_text'] = all_error_text
    dataset['correction'] = all_corrections
    dataset['error_type'] = all_error_type

    return dataset

detected_dataset=dataset_with_errant(dataset)
base_dir = os.path.dirname(os.path.dirname(__file__)) 
save_path = os.path.join(base_dir, "data", "dataset_detected_errant.csv")

# test_dataset = dataset.head(10).copy()
# detected_dataset = dataset_with_errant(test_dataset)
# print(detected_dataset[['text', 'corrected_text', 'error_text', 'correction', 'error_type']])



    # for i in dataset.index:
    #     text = dataset.loc[i, 'text']
    #     matches = annotator.parse(text)
    #     error_text.append(";".join([ text[m.offset : m.offset + m.errorLength] for m in matches]))
    #     error_type.append("; ".join([m.type for m in matches]))
    #     replacements.append([m.replacements for m in matches])
    #     category.append("; ".join([m.category for m in matches]))
    #     if i%1000==0:
    #         print(f"Processed {i}/{len(dataset)}")

    # dataset['error_text']= error_text
    # dataset['error_type'] = error_type
    # dataset['replacements'] = replacements
    # dataset['category'] = category
    # return dataset

# detected_dataset=dataset_detector(dataset)
# base_dir = os.path.dirname(os.path.dirname(__file__)) 
# save_path = os.path.join(base_dir, "data", "dataset_detected.csv")

# detected_dataset.to_csv(save_path, index=False)

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