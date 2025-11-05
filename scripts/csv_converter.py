import pandas as pd
import glob

# # TSV to CSV conversion for C4 dataset parts

# files = sorted(glob.glob("data/C4_200M.tsv-00002-of-00010"))

# output_file = "data/c4_4.csv"

# first = True
# for file in files:
#     print(f"Processing {file}...")
#     reader = pd.read_csv(file, sep='\t', chunksize=10_000, low_memory=False)
#     for chunk in reader:
#         chunk.to_csv(output_file, mode='a', index=False, header=first)
#         first = False

# print("Done:", output_file)

#----------------------------------------------------------------------------#

# Sampling 1k rows from the CSV file

input_file = "data/c4_4.csv"
output_file = "data/c4_4_1k.csv"

sample_df = pd.read_csv(input_file).sample(n=1000, random_state=42)

sample_df.to_csv(output_file, index=False)

print("Done:", output_file)

#----------------------------------------------------------------------------#

# Sampling distribution from the CSV file


