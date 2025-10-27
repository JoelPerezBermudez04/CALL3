import pandas as pd

dataset = pd.read_json('./data/train.jsonl', lines=True)
dataset[['instruction', 'input_text']] = dataset['src'].str.split(pat=':', n=1, expand=True)
dataset = dataset.drop(columns=['src'])
dataset.to_csv('./data/Grammer_Checker.csv', index=False, encoding='utf-8-sig')
print(dataset.head())
