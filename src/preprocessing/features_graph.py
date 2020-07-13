import numpy as np
import pandas as pd
import networkx as nx
import gc
import csv
import os
import pickle
from tqdm import tqdm


def read_header(csv_path: str) -> list:
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
    return header


def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(mn-1, inplace=True)
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100*mem_usg/start_mem_usg, "% of the initial size")
    return props, NAlist


def read_multiple_csv(selected_dir, columns, header=None,
                      total_files=None, dtype=None,
                      usecols=[
                          'engaged_with_user_id_id', 'engaging_user_id_id',
                          'indicator_interaction', 'tweet_id_id']):

    to_concat = []
    cont = 0
    for file in tqdm(os.listdir(selected_dir)):
        if file.endswith('csv'):

            to_concat.append(pd.read_csv(os.path.join(selected_dir, file),
                                         header=header,
                                         names=columns, dtype=dtype,
                                         usecols=usecols))
            if total_files is not None:
                cont += 1
                if cont >= total_files:
                    break
    df = pd.concat(to_concat)
    return df


my_cols = read_header(
    'train-final-complete/partition.csv'
)
train_path = 'train-final-complete'
val_path = 'val-final-complete'

df_train = read_multiple_csv(train_path, columns=my_cols, header=0,
                             total_files=4, dtype=None)

df_train, _ = reduce_mem_usage(df_train)
dtype_dict = df_train.dtypes.to_dict()

df_train = read_multiple_csv(train_path, columns=my_cols, header=0,
                             total_files=None, dtype=dtype_dict)

all_users = df_train["engaged_with_user_id_id"].append(df_train["engaging_user_id_id"]).unique()

map_users = dict(zip(all_users, list(range(len(all_users)))))
df_train["n_engaged_with_id"] = df_train["engaged_with_user_id_id"].map(map_users)
df_train["n_engaging_id"] = df_train["engaging_user_id_id"].map(map_users)


twitter_graph = nx.Graph()
for nodes in tqdm(df_train[df_train["indicator_interaction"] == 1][["n_engaged_with_id",
                  "n_engaging_id"]].itertuples()):
    twitter_graph.add_edge(nodes[1], nodes[2])


adj_set = dict()
for n, nbrdict in tqdm(twitter_graph.adjacency()):
    adj_set[n] = set(nbrdict.keys())

jaccard_features = []
len1 = []
len2 = []
intersec = []
union = []
for i in tqdm(zip(df_train["n_engaged_with_id"],
              df_train["n_engaging_id"])):

    autor_node = i[0]
    engaging_node = i[1]

    if ((autor_node in adj_set) and (engaging_node in adj_set)):
        n1 = adj_set[autor_node]
        n2 = adj_set[engaging_node]
        my_inter = len(n1.intersection(n2))
        my_union = len(n1.union(n2))

        jaccard_features.append(my_inter/my_union)
        intersec.append(my_inter)
        union.append(my_union)
        len1.append(len(n1))
        len2.append(len(n2))

    else:
        jaccard_features.append(-1.0)
        intersec.append(0)

        if autor_node in adj_set:
            my_len1 = len(adj_set[autor_node])
            len1.append(my_len1)
            len2.append(0)
            union.append(my_len1)

        elif engaging_node in adj_set:
            my_len2 = len(adj_set[engaging_node])
            len1.append(0)
            len2.append(my_len2)
            union.append(my_len2)
        else:
            len1.append(0)
            len2.append(0)
            union.append(0)

df_train['jaccard'] = jaccard_features
df_train['len1'] = len1
df_train['len2'] = len2
df_train['union'] = union
df_train['intersec'] = intersec
df_train.to_csv('big_train_features_2.csv', index=False)

del df_train
gc.collect()

df_val = read_multiple_csv(val_path, columns=my_cols, header=0,
                           total_files=None, dtype=dtype_dict)

# mapping as done in training
df_val["n_engaged_with_id"] = df_val["engaged_with_user_id_id"].map(map_users)
df_val["n_engaging_id"] = df_val["engaging_user_id_id"].map(map_users)

# to int
df_val['n_engaged_with_id'] = df_val['n_engaged_with_id'].fillna(-999).astype(int)
df_val['n_engaging_id'] = df_val['n_engaging_id'].fillna(-999).astype(int)


jaccard_features = []
len1 = []
len2 = []
intersec = []
union = []
for i in tqdm(zip(df_val["n_engaged_with_id"],
              df_val["n_engaging_id"])):

    autor_node = i[0]
    engaging_node = i[1]

    if ((autor_node in adj_set) and (engaging_node in adj_set)):
        n1 = adj_set[autor_node]
        n2 = adj_set[engaging_node]
        my_inter = len(n1.intersection(n2))
        my_union = len(n1.union(n2))

        jaccard_features.append(my_inter/my_union)
        intersec.append(my_inter)
        union.append(my_union)
        len1.append(len(n1))
        len2.append(len(n2))

    else:
        jaccard_features.append(-1.0)
        intersec.append(0)

        if autor_node in adj_set:
            my_len1 = len(adj_set[autor_node])
            len1.append(my_len1)
            len2.append(0)
            union.append(my_len1)

        elif engaging_node in adj_set:
            my_len2 = len(adj_set[engaging_node])
            len1.append(0)
            len2.append(my_len2)
            union.append(my_len2)
        else:
            len1.append(0)
            len2.append(0)
            union.append(0)

df_val['jaccard'] = jaccard_features
df_val['len1'] = len1
df_val['len2'] = len2
df_val['union'] = union
df_val['intersec'] = intersec
df_val.to_csv('big_val_features_2.csv', index=False)

my_cols = read_header(
    'submission-final-complete/partition.csv'
)
sub_path = 'submission-final-complete'


df_sub = read_multiple_csv(
    sub_path, columns=my_cols, header=0,
    total_files=4, dtype=None,
    usecols=['engaged_with_user_id_id', 'engaging_user_id_id', 'tweet_id_id']
)

df_sub, _ = reduce_mem_usage(df_sub)
dtype_dict = df_sub.dtypes.to_dict()

df_sub = read_multiple_csv(
    sub_path, columns=my_cols, header=0,
    total_files=None, dtype=dtype_dict,
    usecols=['engaged_with_user_id_id', 'engaging_user_id_id', 'tweet_id_id']
)


# mapping as done in training
df_sub["n_engaged_with_id"] = df_sub["engaged_with_user_id_id"].map(map_users)
df_sub["n_engaging_id"] = df_sub["engaging_user_id_id"].map(map_users)

# to int
df_sub['n_engaged_with_id'] = df_sub['n_engaged_with_id'].fillna(-999).astype(int)
df_sub['n_engaging_id'] = df_sub['n_engaging_id'].fillna(-999).astype(int)

jaccard_features = []
len1 = []
len2 = []
intersec = []
union = []
for i in tqdm(zip(df_sub["n_engaged_with_id"],
              df_sub["n_engaging_id"])):

    autor_node = i[0]
    engaging_node = i[1]

    if ((autor_node in adj_set) and (engaging_node in adj_set)):
        n1 = adj_set[autor_node]
        n2 = adj_set[engaging_node]
        my_inter = len(n1.intersection(n2))
        my_union = len(n1.union(n2))

        jaccard_features.append(my_inter/my_union)
        intersec.append(my_inter)
        union.append(my_union)
        len1.append(len(n1))
        len2.append(len(n2))

    else:
        jaccard_features.append(-1.0)
        intersec.append(0)

        if autor_node in adj_set:
            my_len1 = len(adj_set[autor_node])
            len1.append(my_len1)
            len2.append(0)
            union.append(my_len1)

        elif engaging_node in adj_set:
            my_len2 = len(adj_set[engaging_node])
            len1.append(0)
            len2.append(my_len2)
            union.append(my_len2)
        else:
            len1.append(0)
            len2.append(0)
            union.append(0)

df_sub['jaccard'] = jaccard_features
df_sub['len1'] = len1
df_sub['len2'] = len2
df_sub['union'] = union
df_sub['intersec'] = intersec
df_sub.to_csv('big_sub_features_2.csv', index=False)

# save map_users
with open('map_users.pickle', 'wb') as handle:
    pickle.dump(map_users, handle, protocol=pickle.HIGHEST_PROTOCOL)
