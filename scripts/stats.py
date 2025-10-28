import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)

dataset = pd.read_csv("./data/dataset_detected_errant.csv")

def split(x):
    if isinstance(x, str):
        x = x.strip("[]").replace("'", "")
        return [lang.strip() for lang in x.split(",") if lang.strip()]
    return []

def split_error_types(x):
    if isinstance(x, str) and x.strip():
        return [err.strip() for err in x.split(";")]
    return []

dataset['native_split'] = dataset['native'].apply(split)

dataset['error_type_list'] = dataset['error_type'].apply(split_error_types)

all_langs = [lang for langs in dataset['native_split'] for lang in langs]

lang_counts = pd.Series(all_langs).value_counts()
lang_percentages = (lang_counts / lang_counts.sum() * 100).round(2)

# # Display language's sentence counts
# print(lang_counts[:108])
# print("Unique languages:", len(lang_counts))

#----------------------------------------------------------------------------#

# # Pie Chart Distribution 8 Highest Languages 
# top_n = 8
# top_langs = lang_counts[:top_n]
# other_count = lang_counts[top_n:].sum()
# top_langs['Other'] = other_count
# piechart = top_langs.plot(kind='pie', autopct='%1.1f%%', figsize=(6,6))
# plt.title('Native Language Distribution')
# plt.ylabel('')
# plt.legend(loc='lower right')
# plt.show()

#----------------------------------------------------------------------------#

# # Bar Chart of Top 5 Languages
# top_5_langs = lang_counts.head(5)
# top_5_langs_df = top_5_langs.reset_index()
# top_5_langs_df.columns = ['Language', 'Sentence Count']

# plt.figure(figsize=(10, 6))
# plt.bar(top_5_langs_df['Language'], top_5_langs_df['Sentence Count'], color='skyblue')
# plt.title('Top 5 Native Languages by Sentence Count')
# plt.xlabel('Native Language')
# plt.ylabel('Number of Sentences')
# plt.xticks(rotation=30, ha='right')
# plt.show()

#----------------------------------------------------------------------------#

# # Error Type Distribution by Native Language for Top 5 Languages
# top_5_names = lang_counts.head(5).index.tolist()

# df_exploded_langs = dataset.explode('native_split').copy()
# df_exploded_langs.rename(columns={'native_split': 'Native Language'}, inplace=True)
# df_top_5_langs = df_exploded_langs[df_exploded_langs['Native Language'].isin(top_5_names)].copy()

# df_exploded_errors = df_top_5_langs.explode('error_type_list').copy()
# df_exploded_errors.rename(columns={'error_type_list': 'Error Type'}, inplace=True)
# df_exploded_errors.dropna(subset=['Error Type'], inplace=True)

# error_distribution = df_exploded_errors.groupby(['Native Language', 'Error Type']).size().reset_index(name='Count')

# pivot_table = error_distribution.pivot(index='Native Language', columns='Error Type', values='Count').fillna(0)

# pivot_table['Total'] = pivot_table.sum(axis=1)
# pivot_table = pivot_table.sort_values(by='Total', ascending=False).drop(columns=['Total'])

# plt.figure(figsize=(18, 10))
# ax = pivot_table.plot(
#     kind='bar', 
#     figsize=(18, 10), 
#     width=0.8,
#     colormap='viridis',
#     ax=plt.gca())

# plt.title(
#     f'Error Type Distribution by Native Language for All Error Types ({", ".join(top_5_names)})', 
#     fontsize=18, 
#     pad=20)
# plt.xlabel('Native Language', fontsize=14, labelpad=15)
# plt.ylabel('Total Error Count', fontsize=14, labelpad=15)
# plt.xticks(rotation=0, ha='center', fontsize=12)
# plt.yticks(fontsize=12)
# plt.legend(title='Error Type', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, title_fontsize=12)
# plt.grid(axis='y', linestyle='--', alpha=0.6)
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()

#----------------------------------------------------------------------------#

# # Error Type Distribution CSV (Percentages) for Top 5 Languages
# pivot_table['Total Errors'] = pivot_table.sum(axis=1)

# pivot_table_percentage = pivot_table.div(pivot_table['Total Errors'], axis=0) * 100

# error_percentage_table = pivot_table_percentage.drop(columns=['Total Errors']).round(4)

# error_percentage_table['Total Errors (Count)'] = pivot_table['Total Errors']

# error_percentage_table = error_percentage_table.sort_values(by='Total Errors (Count)', ascending=False)

# output_filename = 'data/error_distribution_top5.csv'
# error_percentage_table.to_csv(output_filename)

# print(f"Error type distribution table (percentages) for top 5 languages saved to '{output_filename}'")

#----------------------------------------------------------------------------#

# # Error Type Distribution by Native Language for Five European Languages

# FIVE_LANGS = ['Spanish', 'French', 'Russian', 'Czech', 'German']

# df_exploded_langs = dataset.explode('native_split').copy()
# df_exploded_langs.rename(columns={'native_split': 'Native Language'}, inplace=True)
# df_five_langs = df_exploded_langs[df_exploded_langs['Native Language'].isin(FIVE_LANGS)].copy()

# df_exploded_errors = df_five_langs.explode('error_type_list').copy()
# df_exploded_errors.rename(columns={'error_type_list': 'Error Type'}, inplace=True)
# df_exploded_errors.dropna(subset=['Error Type'], inplace=True)

# error_distribution = df_exploded_errors.groupby(['Native Language', 'Error Type']).size().reset_index(name='Count')

# pivot_table = error_distribution.pivot(index='Native Language', columns='Error Type', values='Count').fillna(0)

# pivot_table['Total'] = pivot_table.sum(axis=1)
# pivot_table_sorted = pivot_table.sort_values(by='Total', ascending=False).drop(columns=['Total'])

#---------------------------------------------------------------------------#

# # Bar Chart of Error Type Distribution for Five European Languages
# plt.figure(figsize=(18, 10))
# ax = pivot_table_sorted.plot(
#     kind='bar',
#     figsize=(18, 10),
#     width=0.8,
#     colormap='viridis',
#     ax=plt.gca())
# plt.title(
#     f'Error Type Distribution by Native Language for All Error Types ({", ".join(FIVE_LANGS)})',
#     fontsize=18,
#     pad=20)
# plt.xlabel('Native Language', fontsize=14, labelpad=15)
# plt.ylabel('Total Error Count', fontsize=14, labelpad=15)
# plt.xticks(rotation=0, ha='center', fontsize=12)
# plt.yticks(fontsize=12)
# plt.legend(title='Error Type', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, title_fontsize=12)
# plt.grid(axis='y', linestyle='--', alpha=0.6)
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()

#----------------------------------------------------------------------------#

# # Error Type Distribution CSV (Percentages) for Five European Languages
# error_cols = [c for c in pivot_table.columns if c not in ['Total', 'Total Errors']]
# pivot_table['Total Errors'] = pivot_table[error_cols].sum(axis=1)
# pivot_table_percentage = pivot_table[error_cols].div(pivot_table['Total Errors'], axis=0) * 100
# error_percentage_table = pivot_table_percentage.round(4)
# error_percentage_table['Total Errors (Count)'] = pivot_table['Total Errors']
# error_percentage_table = error_percentage_table.sort_values(by='Total Errors (Count)', ascending=False)

# output_filename = 'data/error_distribution_5eu_.csv'
# error_percentage_table.to_csv(output_filename)

#----------------------------------------------------------------------------#

# # Pie Chart Distribution of Spanish

# spanish_df = dataset.explode('native_split')
# spanish_df = spanish_df[spanish_df['native_split'] == 'Spanish'].copy()

# spanish_errors = spanish_df.explode('error_type_list').copy()
# spanish_errors.rename(columns={'error_type_list': 'Error Type'}, inplace=True)
# spanish_errors.dropna(subset=['Error Type'], inplace=True)

# error_counts = spanish_errors['Error Type'].value_counts()

# top_n = 8
# if len(error_counts) > top_n:
#     top_errors = error_counts.head(top_n)
#     other_sum = error_counts[top_n:].sum()
#     top_errors['Other'] = other_sum
# else:
#     top_errors = error_counts

# plt.figure(figsize=(10, 10))

# wedges, texts, autotexts = plt.pie(
#     top_errors,
#     labels=None,
#     autopct='%1.1f%%',
#     startangle=140,
#     pctdistance=0.95
# )
# plt.legend(
#     wedges,
#     top_errors.index,
#     title='Error Type',
#     bbox_to_anchor=(1.05, 1),
#     loc='upper left'
# )
# plt.title('Error Type Distribution for Spanish Speakers')
# plt.tight_layout()
# plt.show()

#----------------------------------------------------------------------------#

# Error Type Distribution Spanish Bar Chart Split by Error (NOT WORKING YET)

# def split_and_strip(x):
#     if isinstance(x, str):
#         return [item.strip() for item in x.split(',') if item.strip()]
#     else:
#         return []

# dataset['nativesplit'] = dataset['native'].apply(split_and_strip)
# spanish_df = dataset.explode('nativesplit')
# spanish_df = spanish_df[spanish_df['nativesplit'] == 'Spanish'].copy()
# spanish_df['errortypelist'] = spanish_df['error_type'].apply(split_and_strip)
# spanish_errors = spanish_df.explode('errortypelist').copy()
# spanish_errors = spanish_errors.dropna(subset=['errortypelist'])

# def base_type(error_code):
#     if ':' in error_code:
#         return error_code.split(':', 1)[1]
#     return error_code

# spanish_errors['BaseType'] = spanish_errors['errortypelist'].map(base_type)
# error_counts = spanish_errors['BaseType'].value_counts()

# plt.figure(figsize=(12, 6))
# error_counts.plot(kind='bar')
# plt.title('Error Types for Spanish Speakers (Grouped by Base Type)')
# plt.xlabel('Error Type')
# plt.ylabel('Count')
# plt.tight_layout()
# plt.show()