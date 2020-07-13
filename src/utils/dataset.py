import tensorflow as tf
import os
from collections import defaultdict
import csv


def build_dataset(filenames, types_list, col_idx, inputs,
                  mode="train", epochs=1, batch_size=512):
    if mode == "train":
        pass
    else:
        pass
    dataset = tf.data.experimental.CsvDataset(
        filenames,
        types_list,
        header=True
    )
    dataset = dataset.map(lambda *x: preprocess(x, col_idx, inputs), num_parallel_calls=-1)
    if mode == "train":
        dataset = dataset.repeat(count=epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset


def get_csvs(path):
    return [os.path.join(path, filename) for filename in os.listdir(path)
            if filename.endswith('.csv')]


def read_header(csv_path: str) -> list:
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
    return header


def create_type_list(file, labels_to_pred):
    types_list = []
    names2idx = dict()
    for i, col in enumerate(read_header(file)):
        if col.endswith('_ors'):
            types_list.append(tf.string)
        elif col.endswith('_id'):
            types_list.append(tf.string)
        elif col.endswith('_cat') and col != 'cluster_cat':
            types_list.append(tf.float32)
        elif col.endswith('_bool'):
            types_list.append(tf.string)
        elif col.endswith('_num'):
            types_list.append(tf.float32)
        elif col == 'hashtagEncoded_unors':
            types_list.append(tf.string)
        elif col == 'domainEncoded_unors':
            types_list.append(tf.string)
        elif col.startswith('indicator') and (col in labels_to_pred):
            types_list.append(tf.int32)
        elif col.endswith('bucket') or col.endswith('bucket_2'):
            types_list.append(tf.int32)
        elif col.endswith('_topicprop'):
            types_list.append(tf.float32)
        elif col.endswith('_topiccount'):
            types_list.append(tf.float32)
        elif col.endswith('_topic'):
            types_list.append(tf.int32)
        elif col == 'cluster_cat':
            types_list.append(tf.int32)
        elif col.endswith('ss'):
            types_list.append(tf.float32)
        elif col == 'engaging_follows_engaged':
            types_list.append(tf.int32)
        else:
            types_list.append(tf.string)
        names2idx[i] = col
    return types_list, names2idx


def preprocess(x, dict_idx, inputs):
    """
    Only tf operations
    """
    return_dict = {}
    labels = {}
    for key, indexes in dict_idx.items():
        if key == 'lab':
            pass
        elif key in inputs:
            return_dict[key] = []
        else:
            continue
        for idx in indexes:
            if key == "txt":
                bert_encoding = tf.strings.regex_replace(
                    tf.strings.regex_replace(x[idx], "\[", ""), "\]", "")
                bert_encoding = tf.strings.split(bert_encoding, sep=',')
                return_dict[key] = tf.strings.to_number(bert_encoding, tf.float32)  # bert_encoding
            elif key in ['author_bool_cols', 'engager_bool_cols',
                         'interaction_bool_cols']:
                result = tf.cond((x[idx] == "true") or (x[idx] == "True"),
                                 lambda: 1.0, lambda: 0.0)
                return_dict[key].append(result)
            elif key in ['tweet_num_cols', 'author_num_cols',
                         'engager_num_cols', 'interaction_num_cols',
                         "engager_hist_num_cols"]:
                return_dict[key].append(x[idx])
            elif key in ['tweet_cat_cols', 'author_cat_cols',
                         'engager_cat_cols', 'engager_topic_cat']:
                return_dict[key].append(tf.cast(x[idx], tf.int32))
            elif key == 'bucket':
                return_dict[key].append(x[idx])
            elif key == 'hsh':
                hsh_encode = tf.strings.regex_replace(
                    tf.strings.regex_replace(x[idx], "\[", ""), "\]", "")
                hsh_encode = tf.strings.split(hsh_encode, sep=',', maxsplit=6)[:5]
                hsh_encode = tf.strings.to_number(hsh_encode, tf.int32)  # bert_encoding
                length = tf.shape(hsh_encode)[0]
                paddings = [[0, 5-length]]
                hsh_encode = tf.pad(hsh_encode, paddings, 'CONSTANT', constant_values=0)
                return_dict[key] = hsh_encode
            elif key == 'dom':
                dom_encode = tf.strings.regex_replace(
                    tf.strings.regex_replace(x[idx], "\[", ""), "\]", "")
                dom_encode = tf.strings.split(dom_encode, sep=',', maxsplit=6)[:5]
                dom_encode = tf.strings.to_number(dom_encode, tf.int32)  # bert_encoding
                length = tf.shape(dom_encode)[0]
                paddings = [[0, 5-length]]
                dom_encode = tf.pad(dom_encode, paddings, 'CONSTANT', constant_values=0)
                return_dict[key] = dom_encode
            elif key == 'id_cols':
                return_dict[key].append(x[idx])
            elif key == 'lab':
                labels[idx] = [x[dict_idx[key][idx]]]
            elif key == 'engager_topic_count':
                return_dict[key].append(x[idx])
            elif key == 'engager_topic_prop':
                return_dict[key].append(x[idx])
            elif key == 'bucket_2':
                return_dict[key].append(x[idx])
            elif key == 'new_features':
                return_dict[key].append(x[idx])
            elif key == 'engaging_follows_engaged':
                return_dict[key].append(x[idx])
    return return_dict, labels


def names_to_idx(csv_path: str, labels_to_pred) -> defaultdict:
    id_cols = ['tweet_id_id', 'engaged_with_user_id_id',
               'engaging_user_id_id']
    tweet_num_cols = ['hashtagSumCount_ss_num', 'hashtagCount_ss_num',
                      'domainCount_ss_num', 'PhotoCount_ss_num',
                      'VideoCount_ss_num', 'GIFCount_ss_num',
                      'linkCount_ss_num']
    tweet_cat_cols = ['tweetEncoded_cat', 'languageEncoded_cat',
                      'tweet_timestamp_day_of_week_cat',
                      'tweet_timestamp_hour_cat', 'cluster_cat']
    author_num_cols = ['engaged_follow_diff_log_ss_num',
                       'engaged_with_user_follower_count_log_ss_num',
                       'engaged_with_user_following_count_log_ss_num']
    author_bool_cols = ['engaged_with_user_is_verified_bool']
    author_cat_cols = ['engaged_with_user_account_creation_q_cat']
    engager_num_cols = ['engaging_follow_diff_log_ss_num',
                        'engaging_user_follower_count_log_ss_num',
                        'engaging_user_following_count_log_ss_num']
    engager_bool_cols = ['engaging_user_is_verified_bool']
    engager_cat_cols = ['engaging_user_account_creation_q_cat']
    interaction_num_cols = ['engaged_follower_diff_engaging_following_log_ss_num',
                            'engaged_following_diff_engaging_follower_log_ss_num',
                            'engaged_with_vs_engaging_follower_diff_log_ss_num',
                            'engaged_with_vs_engaging_following_diff_log_ss_num']
    interaction_bool_cols = ['engagee_follows_engager_bool']
    engager_topic_cat = ['0_topic', '1_topic', '2_topic', '3_topic', '4_topic']
    engager_topic_count = ['0_topiccount', '1_topiccount', '2_topiccount',
                           '3_topiccount', '4_topiccount']
    new_features = ['total_appearance_ss_num', 'n_appearances_author_ss',
                    'n_tweets_ss', 'total_count_couple_ss', 'jaccard_ss',
                    'len1_ss', 'len2_ss', 'union_ss', 'intersec_ss']
    engaging_follows_engaged = ["engaging_follows_engaged"]

    col_to_idx = defaultdict(list)
    header = read_header(csv_path)
    col_to_idx['lab'] = dict()
    for i, col in enumerate(header):
        if col in id_cols:
            col_to_idx['id_cols'].append(i)
        elif col in tweet_num_cols:
            col_to_idx['tweet_num_cols'].append(i)
        elif col in tweet_cat_cols:
            col_to_idx['tweet_cat_cols'].append(i)
        elif col in author_num_cols:
            col_to_idx['author_num_cols'].append(i)
        elif col in author_bool_cols:
            col_to_idx['author_bool_cols'].append(i)
        elif col in author_cat_cols:
            col_to_idx['author_cat_cols'].append(i)
        elif col in engager_num_cols:
            col_to_idx['engager_num_cols'].append(i)
        elif col in engager_bool_cols:
            col_to_idx['engager_bool_cols'].append(i)
        elif col in engager_cat_cols:
            col_to_idx['engager_cat_cols'].append(i)
        elif col in interaction_num_cols:
            col_to_idx['interaction_num_cols'].append(i)
        elif col in interaction_bool_cols:
            col_to_idx['interaction_bool_cols'].append(i)
        elif col in engager_topic_cat:
            col_to_idx['engager_topic_cat'].append(i)
        elif col in engager_topic_count:
            col_to_idx['engager_topic_count'].append(i)
        elif col == 'embedding_ors':
            col_to_idx['txt'].append(i)
        elif col == 'hashtagEncoded_unors':
            col_to_idx['hsh'].append(i)
        elif col == 'domainEncoded_unors':
            col_to_idx['dom'].append(i)
        elif col.startswith('indicator') and (col in labels_to_pred):
            col_to_idx['lab'][col] = i
        elif col.endswith('bucket'):
            col_to_idx['bucket'].append(i)
        elif col.endswith('bucket_2'):
            col_to_idx['bucket_2'].append(i)
        elif col in new_features:
            col_to_idx['new_features'].append(i)
        elif col in engaging_follows_engaged:
            col_to_idx['engaging_follows_engaged'].append(i)
    return col_to_idx
