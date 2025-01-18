import cProfile
import pathlib
import pstats

import pandas as pd


create_tf_idf_table_to_path = f"{pathlib.Path().resolve()}\\TF_IDF_V2_Table.parquet"

df = pd.read_parquet(create_tf_idf_table_to_path)

print(df.shape[0])

print(df)

np_df = df.reset_index().to_numpy()

print(np_df)

tf_idf_table = pd.read_parquet(create_tf_idf_table_to_path).reset_index().to_numpy()

print("--------------------------------------------")

print(np_df[0][0])
print(type(np_df[0][0]))

print(tf_idf_table[int(np_df[0][0])][1:-1])

# print(tf_idf_table[np_df[0][0]][1:-1])

train_path = f"{pathlib.Path().resolve()}\\folds\\0\\train0.csv"
test_path = f"{pathlib.Path().resolve()}\\folds\\0\\test0.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

from knn import calculate_distances_between_test_and_train, calculate_distances_between_test_and_train_mt

def test():
    distances = calculate_distances_between_test_and_train(train_data, test_data, create_tf_idf_table_to_path)
    print(distances)

cProfile.run('test()', 'profile_output')
p = pstats.Stats('profile_output')
p.sort_stats(pstats.SortKey.TIME).print_stats(10)
#
