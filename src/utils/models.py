import tensorflow as tf
from utils.layers import FM_pro, EncoderLayer


def build_inputs(hparams):
    total_inputs = {
        'txt': tf.keras.Input(
            name='txt',
            shape=(768,),
            dtype='float32'
        ),
        'id_cols': tf.keras.Input(
            name='id_cols',
            shape=(3,),
            dtype='string'
        ),
        'author_bool_cols': tf.keras.Input(
            name='author_bool_cols',
            shape=(1,),
            dtype='float32'
        ),
        'engager_bool_cols': tf.keras.Input(
            name='engager_bool_cols',
            shape=(1,),
            dtype='float32'
        ),
        'interaction_bool_cols': tf.keras.Input(
            name='interaction_bool_cols',
            shape=(1,),
            dtype='float32'
        ),
        'hsh': tf.keras.Input(
            name='hsh',
            shape=(5,),
            dtype='int32'
        ),
        'tweet_num_cols': tf.keras.Input(
            name='tweet_num_cols',
            shape=(7,),
            dtype='float32'
        ),
        'dom': tf.keras.Input(
            name='dom',
            shape=(5,),
            dtype='int32'
        ),
        'tweet_cat_cols': tf.keras.Input(
            name='tweet_cat_cols',
            shape=(5,),
            dtype='int32'
        ),
        'author_num_cols': tf.keras.Input(
            name='author_num_cols',
            shape=(3,),
            dtype='float32'
        ),
        'engager_num_cols': tf.keras.Input(
            name='engager_num_cols',
            shape=(3,),
            dtype='float32'
        ),
        'interaction_num_cols': tf.keras.Input(
            name='interaction_num_cols',
            shape=(4,),
            dtype='float32'
        ),
        'author_cat_cols': tf.keras.Input(
            name='author_cat_cols',
            shape=(1,),
            dtype='int32'
        ),
        'engager_cat_cols': tf.keras.Input(
            name='engager_cat_cols',
            shape=(1,),
            dtype='int32'
        ),
        'bucket': tf.keras.Input(
            name='bucket',
            shape=(2,),
            dtype='int32'
        ),
        'engager_topic_cat': tf.keras.Input(
            name='engager_topic_cat',
            shape=(5,),
            dtype='int32'
        ),
        'engager_topic_count': tf.keras.Input(
            name='engager_topic_count',
            shape=(5,),
            dtype='float32'
        ),
        'new_features': tf.keras.Input(
            name="new_features",
            shape=(9,),
            dtype="float32"
        ),
        'engaging_follows_engaged': tf.keras.Input(
            name="engaging_follows_engaged",
            shape=(1,),
            dtype="int32"
        ),
        "bucket_2": tf.keras.Input(
            name='bucket_2',
            shape=(2,),
            dtype='int32'
        ),
    }
    inputs = {}
    used_inputs = hparams["inputs"]
    for key, value in total_inputs.items():
        if key in used_inputs:
            inputs[key] = value
    return inputs


def final_model_best_50_0(hparams):
    """
    Final model with Dense for all features
    """
    labels_to_pred = hparams["labels"].keys()
    metadata = hparams["metadata"]
    inputs = build_inputs(hparams)
    params = hparams["model_params"]
    ######### CREATING LAYERS #############
    # TEXT
    encoder_txt = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                params["embedding_users_dim"]*2,
                activation="relu"
            ),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(params["embedding_users_dim"], activation="relu")
        ],
        name="encoder_txt"
    )
    # USER
    emb_user_id = tf.keras.layers.Embedding(
        metadata["num_user_buckets"],
        params["embedding_users_dim"], name="emb_user_id"
    )

    # ENGAGED USER
    extract_engaged = tf.keras.layers.Lambda(lambda x: x[:, 0], name="extract_engaged")
    engaged_num_bools = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(params["embedding_users_dim"]//4),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(params["dr_num_bool"]),
            tf.keras.layers.Dense(
                params["embedding_users_dim"]//4),  # //2
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(params["dr_num_bool"])
        ],
        name="engaged_num_bools"
    )
    emb_engaged_account_creation = tf.keras.layers.Embedding(
        metadata["author_cat_cols"]["engaged_with_user_account_creation_q_cat"],
        params["embedding_users_dim"]//8,
        name="emb_engaged_account_creation"
    )

    # ENGAGING USER ID
    extract_engaging = tf.keras.layers.Lambda(lambda x: x[:, 1], name="extract_engaging")
    engaging_num_bools = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(params["embedding_users_dim"]//4),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(params["dr_num_bool"]),
            tf.keras.layers.Dense(
                params["embedding_users_dim"]//4),  # //2
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(params["dr_num_bool"])
        ],
        name="engaging_num_bools"
    )
    emb_engaging_account_creation = tf.keras.layers.Embedding(
        metadata["engager_cat_cols"]["engaging_user_account_creation_q_cat"],
        params["embedding_users_dim"]//8,
        name="emb_engaging_account_creation"
    )

    # TWEET
    tweet_num = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(params["embedding_users_dim"]//4),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(params["dr_num_bool"]),
            tf.keras.layers.Dense(params["embedding_users_dim"]//4),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(params["dr_num_bool"])
        ],
        name="tweet_num"
    )
    extract_tweet_type = tf.keras.layers.Lambda(lambda x: x[:, 0], name="extract_tweet_type")
    extract_language_type = tf.keras.layers.Lambda(lambda x: x[:, 1], name="extract_language_type")
    extract_day_of_week = tf.keras.layers.Lambda(lambda x: x[:, 2], name="extract_day_of_week")
    extract_hour = tf.keras.layers.Lambda(lambda x: x[:, 3], name="extract_hour")
    extract_cluster_topic = tf.keras.layers.Lambda(lambda x: x[:, 4], name="extract_cluster_topic")
    emb_tweet_type = tf.keras.layers.Embedding(
        metadata["tweet_cat_cols"]["tweetEncoded_cat"],
        params["embedding_users_dim"]//8, name="emb_tweet_type"
    )  # Dim=16 # //4
    emb_language_type = tf.keras.layers.Embedding(
        metadata["tweet_cat_cols"]["languageEncoded_cat"],
        params["embedding_users_dim"]//16,
        name="emb_language_type"
    )  # // 8
    emb_day_of_week = tf.keras.layers.Embedding(
        metadata["tweet_cat_cols"]["tweet_timestamp_day_of_week_cat"],
        params["embedding_users_dim"]//16,
        name="emb_day_of_week"
    )  # // 8
    emb_hour = tf.keras.layers.Embedding(
        metadata["tweet_cat_cols"]["tweet_timestamp_hour_cat"],
        params["embedding_users_dim"]//16,
        name="emb_hour"
    )  # // 8
    emb_cluster_topic = tf.keras.layers.Embedding(
        metadata["tweet_cat_cols"]["cluster_cat"],
        params["embedding_users_dim"]//4,
        name="emb_cluster_topic"
    )  # // 2

    # TWEET hsh-dom
    emb_dom = tf.keras.layers.Embedding(
        metadata["domain_size"],
        params["embedding_users_dim"]//8,
        name="emb_dom"
    )  # // 2
    emb_hsh = tf.keras.layers.Embedding(
        metadata["hashtag_size"],
        params["embedding_users_dim"]//8,
        name="emb_hsh"
    )  # // 2

    # INTERACTION
    extract_len_intersect = tf.keras.layers.Lambda(
        lambda x: x[:, 8:9],
        name="extract_len_intersect"
    )
    emb_engaging_follows_engager = tf.keras.layers.Embedding(
        3,
        params["embedding_users_dim"]//16,
        name="emb_engaging_follows_engager"
    )
    interaction_num_bools = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(params["embedding_users_dim"]//4),  # // 2
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(params["dr_num_bool"]),
            tf.keras.layers.Dense(
                params["embedding_users_dim"]//4),  # // 2
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(params["dr_num_bool"])
        ],
        name="interaction_num_bools"
    )

    # TOPICS
    normalizer_topic_count = tf.keras.layers.BatchNormalization(axis=-1)

    # COMBINED
    ff_all = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(params["embedding_users_dim"]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(params["dr_combined"]),
            tf.keras.layers.Dense(params["embedding_users_dim"]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu")
        ],
        name="ff_all"
    )
    ff_user = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(params["embedding_users_dim"]*2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(params["dr_combined"]),
            tf.keras.layers.Dense(params["embedding_users_dim"]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu")
        ],
        name="ff_user"
    )
    ff_tweet = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(params["embedding_users_dim"]*2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(params["dr_combined"]),
            tf.keras.layers.Dense(params["embedding_users_dim"]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu")
        ],
        name="ff_tweet"
    )
    ff_others = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(params["embedding_users_dim"]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(params["dr_combined"]),
            tf.keras.layers.Dense(params["embedding_users_dim"]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu")
        ],
        name="ff_others"
    )
    ff_topics = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(params["embedding_users_dim"]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(params["dr_combined"]),
            tf.keras.layers.Dense(params["embedding_users_dim"]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu")
        ],
        name="ff_topics"
    )
    # ATTENTION
    encoder_attention_0 = EncoderLayer(
        d_model=params["embedding_users_dim"],
        num_heads=4,
        dff=params["embedding_users_dim"]*2,
        rate=0.15,
        normalization='layer',
        attention='dot',
        name="encoder_attention_0"
    )
    encoder_attention_1 = EncoderLayer(
        d_model=params["embedding_users_dim"],
        num_heads=4,
        dff=params["embedding_users_dim"]*2,
        rate=0.15,
        normalization='layer',
        attention='dot',
        name="encoder_attention_1"
    )
    ff_attention = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(params["embedding_users_dim"]*2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(params["dr_dl"]),
            tf.keras.layers.Dense(params["embedding_users_dim"], activation="relu")
        ],
        name="ff_attention"
    )

    # FM
    fm_pro = FM_pro(9, name="fm_pro")

    # DL
    ff_dl = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(params["embedding_users_dim"]*2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(params["dr_dl"]),
            tf.keras.layers.Dense(params["embedding_users_dim"]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(params["dr_dl"]),
            tf.keras.layers.Dense(params["embedding_users_dim"]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu")
        ],
        name="ff_dl"
    )
    # rp
    rp_mlp = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(params["embedding_users_dim"]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(params["dr_dl"]),
            tf.keras.layers.Dense(params["embedding_users_dim"]//2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu")
        ],
        name="rp_mlp"
    )
    # rt & rtc
    rt_rtc_mlp = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(params["embedding_users_dim"]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(params["dr_dl"]),
            tf.keras.layers.Dense(params["embedding_users_dim"]//2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu")
        ],
        name="rt_rtc_mlp"
    )
    # like
    lk_mlp = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(params["embedding_users_dim"]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(params["dr_dl"]),
            tf.keras.layers.Dense(params["embedding_users_dim"]//2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu")
        ],
        name="lk_mlp"
    )

    # OUTPUT
    dict_outputs = {}
    for label in labels_to_pred:
        dict_outputs[label] = tf.keras.layers.Dense(1, activation="sigmoid", name=label)

    ######### FORWARD #############
    # TXT
    txt_enc = encoder_txt(inputs["txt"])  # FINAL
    # AUTHOR
    author_user = extract_engaged(inputs["bucket_2"])
    author_user = emb_user_id(author_user)  # FINAL
    author_num_bool = tf.keras.layers.Concatenate(axis=-1)([inputs["author_bool_cols"],
                                                            inputs["author_num_cols"]])
    author_num_bool = engaged_num_bools(author_num_bool)  # FINAL
    author_acc_creation = emb_engaged_account_creation(inputs["author_cat_cols"])  # FINAL
    # ENGAGING
    engaging_user = extract_engaging(inputs["bucket_2"])
    engaging_user = emb_user_id(engaging_user)  # FINAL
    engaging_num_bool = tf.keras.layers.Concatenate(axis=-1)([inputs["engager_bool_cols"],
                                                              inputs["engager_num_cols"]])
    engaging_num_bool = engaging_num_bools(engaging_num_bool)  # FINAL
    engaging_acc_creation = emb_engaging_account_creation(inputs["engager_cat_cols"])  # FINAL
    # INTERACTION
    n_tweets_author = extract_n_tweets_author(inputs["new_features"])
    len_intersect = extract_len_intersect(inputs["new_features"])
    engaging_follows_engaged = emb_engaging_follows_engager(inputs["engaging_follows_engaged"])
    engaging_follows_engaged = tf.keras.layers.Flatten(
        name="flatten_engaging_follows_engaged")(engaging_follows_engaged)
    interaction_num_bool = tf.keras.layers.Concatenate(axis=-1)([inputs["interaction_bool_cols"],
                                                                 inputs["interaction_num_cols"],
                                                                 engaging_follows_engaged,
                                                                 len_intersect,
                                                                 n_tweets_author])
    interaction_num_bool = interaction_num_bools(interaction_num_bool)  # FINAL
    # TWEET
    tweet_numeric = tweet_num(inputs["tweet_num_cols"])  # FINAL
    tweet_type = extract_tweet_type(inputs["tweet_cat_cols"])
    tweet_type = emb_tweet_type(tweet_type)  # FINAL
    language_type = extract_language_type(inputs["tweet_cat_cols"])
    language_type = emb_language_type(language_type)  # FINAL
    day_of_week = extract_day_of_week(inputs["tweet_cat_cols"])
    day_of_week = emb_day_of_week(day_of_week)  # FINAL
    hour = extract_hour(inputs["tweet_cat_cols"])
    hour = emb_hour(hour)  # FINAL
    cluster_topic = extract_cluster_topic(inputs["tweet_cat_cols"])
    cluster_topic = emb_cluster_topic(cluster_topic)
    cluster_topic = tf.expand_dims(cluster_topic, axis=1)  # FINAL (B, 1, 32)
    # TOPICS ENGAGING
    engager_topics = emb_cluster_topic(inputs["engager_topic_cat"])  # (B, 5, 64)
    topic_count = normalizer_topic_count(tf.expand_dims(inputs["engager_topic_count"], axis=2))
    topics_engaging_counted = tf.multiply(engager_topics, topic_count)  # (B, 5, 64)
    topics_engaging_counted = tf.keras.layers.Flatten()(topics_engaging_counted)

    # COMBINED
    engaging_acc_creation = tf.keras.layers.Flatten()(engaging_acc_creation)
    features_engaging_user = tf.keras.layers.Concatenate()([engaging_user, engaging_num_bool,
                                                            engaging_acc_creation])
    repr_engaging_user_id = ff_user(features_engaging_user)  # Representation (B, 64)

    author_acc_creation = tf.keras.layers.Flatten()(author_acc_creation)
    features_author_user = tf.keras.layers.Concatenate()([author_user, author_num_bool,
                                                          author_acc_creation])
    repr_author_user_id = ff_user(features_author_user)  # Representation (B, 64)

    repr_topics = ff_topics(topics_engaging_counted)

    cluster_topic_2 = tf.keras.layers.Flatten()(cluster_topic)
    # Combined hsh-dom tweet
    hsh_embeddings = emb_hsh(inputs["hsh"])  # B, 5,64
    dom_embeddings = emb_dom(inputs["dom"])  # B,5,64
    important_hsh = tf.keras.layers.GlobalMaxPool1D()(hsh_embeddings)
    important_dom = tf.keras.layers.GlobalMaxPool1D()(dom_embeddings)

    features_tweet = tf.keras.layers.Concatenate()([tweet_numeric, tweet_type,
                                                    language_type, cluster_topic_2,
                                                    important_hsh, important_dom])
    repr_tweet = ff_tweet(features_tweet)  # Representation (B, 64)

    features_others = tf.keras.layers.Concatenate()([interaction_num_bool, day_of_week,
                                                     hour])
    repr_others = ff_others(features_others)  # Representation (B, 64)

    # All
    all_feat = tf.keras.layers.Concatenate()([
        author_num_bool, author_acc_creation, engaging_num_bool,
        engaging_acc_creation, engaging_follows_engaged, tweet_type,
        language_type, day_of_week, hour, important_hsh, important_dom,
        inputs["tweet_num_cols"],  inputs["interaction_bool_cols"], inputs["interaction_num_cols"]
    ])
    repr_all = ff_all(all_feat)

    # ATTENTION
    attention_concatenated = tf.stack([
        repr_author_user_id, repr_tweet, repr_topics, repr_others, repr_all,
        txt_enc, repr_engaging_user_id, author_user, engaging_user], axis=1)  # (B, 5, 128)
    attention_output = encoder_attention_0(attention_concatenated, mask=None)
    attention_output = encoder_attention_1(attention_output, mask=None)  # (B, 5, 128)
    attention_output_max = tf.keras.layers.GlobalMaxPooling1D()(attention_output)
    attention_output = tf.keras.layers.Flatten()(attention_output)  # (B, 320)
    attention_output = ff_attention(attention_output)  # (B, 128)

    # FM
    fm_concatenated = tf.stack([
        repr_engaging_user_id, repr_author_user_id, repr_tweet, repr_topics, repr_all,
        repr_others, txt_enc, author_user, engaging_user], axis=1)
    fm_output = fm_pro(fm_concatenated)  # (B, 17)

    # DL
    df_concatenated = tf.keras.layers.Concatenate()([
        repr_engaging_user_id, repr_author_user_id, repr_tweet, repr_topics, repr_all,
        repr_others, txt_enc, author_user, engaging_user])
    dl_output = ff_dl(df_concatenated)  # (B, 128)

    # rtc and reply & like and retweet
    current_output = tf.keras.layers.Concatenate()([attention_output, fm_output,
                                                    dl_output, attention_output_max])
    rp_output = rp_mlp(current_output)
    rt_rtc_output = rt_rtc_mlp(current_output)
    lk_output = lk_mlp(current_output)

    # OUTPUTS
    outputs = dict()
    for label in labels_to_pred:
        if label == "indicator_reply":
            outputs[label] = dict_outputs[label](rp_output)
        elif label in ["indicator_retweet_with_comment", "indicator_retweet"]:
            outputs[label] = dict_outputs[label](rt_rtc_output)
        elif label == "indicator_like":
            outputs[label] = dict_outputs[label](lk_output)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
