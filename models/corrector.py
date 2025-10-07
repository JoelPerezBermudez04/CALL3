import language_tool_python
import pandas as pd
import os

base_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(base_dir, "data")
input_path = os.path.join(data_dir, "dataset_detected.csv")
output_path = os.path.join(data_dir, "dataset_corrected.csv")

try:
    tool = language_tool_python.LanguageTool('en-GB')
except Exception:
    tool = language_tool_python.LanguageToolPublicAPI('en-GB')

dataset = pd.read_csv('./data/dataset_detected.csv')

def correct_text(text):
    try:
        matches = tool.check(text)
        corrected = language_tool_python.utils.correct(text, matches)
        return corrected
    except Exception as e:
        print(f"Error correcting text: {e}")
        return text
    

def dataset_corrector(dataset):
    corrected_texts = []
    for i in dataset.index:
        text = dataset.loc[i, 'text']
        corrected = correct_text(text)
        corrected_texts.append(corrected)
        if i % 500 == 0:
            print(f"Processed {i}/{len(dataset)}")

    dataset['corrected_text'] = corrected_texts
    return dataset

corrected_dataset = dataset_corrector(dataset)

base_dir = os.path.dirname(os.path.dirname(__file__))
save_path = os.path.join(base_dir, "data", "dataset_corrected.csv")
corrected_dataset.to_csv(save_path, index=False)