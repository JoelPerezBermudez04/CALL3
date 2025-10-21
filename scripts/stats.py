import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)

dataset = pd.read_csv("./data/dataset_corrected.csv")

def split(x):
    if isinstance(x, str):
        x = x.strip("[]").replace("'", "")
        return [lang.strip() for lang in x.split(",") if lang.strip()]
    return []

def split_categories(x):
    if isinstance(x, str) and x.strip():
        return [cat.strip() for cat in x.split(";")]
    return []

dataset['native_split'] = dataset['native'].apply(split)

dataset['category_list'] = dataset['category'].apply(split_categories)

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

# # Error Distribution by Native Language for Top 5 Languages
# top_5_names = lang_counts.head(5).index.tolist()

# df_exploded_langs = dataset.explode('native_split').copy()
# df_exploded_langs.rename(columns={'native_split': 'Native Language'}, inplace=True)
# df_top_5_langs = df_exploded_langs[df_exploded_langs['Native Language'].isin(top_5_names)].copy()

# df_exploded_errors = df_top_5_langs.explode('category_list').copy()
# df_exploded_errors.rename(columns={'category_list': 'Error Category'}, inplace=True)
# df_exploded_errors.dropna(subset=['Error Category'], inplace=True)

# error_distribution = df_exploded_errors.groupby(['Native Language', 'Error Category']).size().reset_index(name='Count')

# pivot_table = error_distribution.pivot(index='Native Language', columns='Error Category', values='Count').fillna(0)

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
#     f'Error Distribution by Native Language for All Categories ({", ".join(top_5_names)})', 
#     fontsize=18, 
#     pad=20)
# plt.xlabel('Native Language', fontsize=14, labelpad=15)
# plt.ylabel('Total Error Count', fontsize=14, labelpad=15)
# plt.xticks(rotation=0, ha='center', fontsize=12)
# plt.yticks(fontsize=12)
# plt.legend(title='Error Category', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, title_fontsize=12)
# plt.grid(axis='y', linestyle='--', alpha=0.6)
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()

# #----------------------------------------------------------------------------#

# # Error Distribution CSV (Percentages) for Top 5 Languages
# pivot_table['Total Errors'] = pivot_table.sum(axis=1)

# pivot_table_percentage = pivot_table.div(pivot_table['Total Errors'], axis=0) * 100

# error_percentage_table = pivot_table_percentage.drop(columns=['Total Errors']).round(4)

# error_percentage_table['Total Errors (Count)'] = pivot_table['Total Errors']

# error_percentage_table = error_percentage_table.sort_values(by='Total Errors (Count)', ascending=False)

# output_filename = 'data/error_distribution.csv'
# error_percentage_table.to_csv(output_filename)

# print(f"Error distribution table (percentages) for top 5 languages saved to '{output_filename}'")

#----------------------------------------------------------------------------#

# Error Distribution by Native Language for Five European Languages

FIVE_LANGS = ['Spanish', 'French', 'Russian', 'Czech', 'German']

df_exploded_langs = dataset.explode('native_split').copy()
df_exploded_langs.rename(columns={'native_split': 'Native Language'}, inplace=True)
df_five_langs = df_exploded_langs[df_exploded_langs['Native Language'].isin(FIVE_LANGS)].copy()

df_exploded_errors = df_five_langs.explode('category_list').copy()
df_exploded_errors.rename(columns={'category_list': 'Error Category'}, inplace=True)
df_exploded_errors.dropna(subset=['Error Category'], inplace=True)

error_distribution = df_exploded_errors.groupby(['Native Language', 'Error Category']).size().reset_index(name='Count')

pivot_table = error_distribution.pivot(index='Native Language', columns='Error Category', values='Count').fillna(0)

pivot_table['Total'] = pivot_table.sum(axis=1)
pivot_table_sorted = pivot_table.sort_values(by='Total', ascending=False).drop(columns=['Total'])

#---------------------------------------------------------------------------#

# Bar Chart of Error Distribution for Five European Languages
plt.figure(figsize=(18, 10))
ax = pivot_table_sorted.plot(
    kind='bar',
    figsize=(18, 10),
    width=0.8,
    colormap='viridis',
    ax=plt.gca())
plt.title(
    f'Error Distribution by Native Language for All Categories ({", ".join(FIVE_LANGS)})',
    fontsize=18,
    pad=20)
plt.xlabel('Native Language', fontsize=14, labelpad=15)
plt.ylabel('Total Error Count', fontsize=14, labelpad=15)
plt.xticks(rotation=0, ha='center', fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Error Category', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, title_fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#----------------------------------------------------------------------------#

# Error Distribution CSV (Percentages) for Five European Languages
pivot_table['Total Errors'] = pivot_table.sum(axis=1)
pivot_table_percentage = pivot_table.div(pivot_table['Total Errors'], axis=0) * 100
error_percentage_table = pivot_table_percentage.drop(columns=['Total Errors']).round(4)
error_percentage_table['Total Errors (Count)'] = pivot_table['Total Errors']
error_percentage_table = error_percentage_table.sort_values(by='Total Errors (Count)', ascending=False)

output_filename = 'data/error_distribution_5_langs.csv'
error_percentage_table.to_csv(output_filename)
