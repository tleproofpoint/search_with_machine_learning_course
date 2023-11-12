import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv
import re
from nltk.stem import *

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/fasttext/labeled_queries.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
queries_df = pd.read_csv(queries_file_name)[['category', 'query']]
queries_df = queries_df[queries_df['category'].isin(categories)].sort_values(by=['category'])

# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
for index, row in queries_df.iterrows():
    query = row['query']
    query_st = re.sub(r'[\W_]+', ' ', query.lower())
    singles = [stemmer.stem(word) for word in query_st.split()]
    row['query'] = ' '.join(singles)

# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.
while True:
    queries_df_grouped = queries_df.groupby('category', as_index = False, sort = True).count()
    filtered_queries_df_grouped = queries_df_grouped[queries_df_grouped['query'] < 1000]
    merged_filtered_queries_df_grouped = pd.merge(filtered_queries_df_grouped, parents_df, on='category')
    size = merged_filtered_queries_df_grouped.size
    if size <= 1:
        break
    print('Size: ' + str(size))
    min_category = merged_filtered_queries_df_grouped.iloc[0]['category']
    max_category = merged_filtered_queries_df_grouped.iloc[-1]['category']
    print(min_category + " -- " + max_category)
    count = 0
    for index, query_row in queries_df.iterrows():
        category = query_row['category']
        if (category < min_category) or (category > max_category):
            continue
        match_df = merged_filtered_queries_df_grouped[merged_filtered_queries_df_grouped['category'] == category]
        if match_df.size > 0:
            parent = match_df.iloc[0]['parent']
            if category != parent:
                query_row['category'] = parent
                count += 1
    if count == 0:
        break

# Create labels in fastText format.
queries_df['label'] = '__label__' + queries_df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
queries_df = queries_df[queries_df['category'].isin(categories)]
queries_df['output'] = queries_df['label'] + ' ' + queries_df['query']
queries_df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)