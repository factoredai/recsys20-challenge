import os
import pyspark
import pyspark.sql.functions as f

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.ml.linalg import Vectors, VectorUDT

from ..eda.utils import parse_data


def AutoPCA(df, inputCol='scaled_embedding', outputCol='pcs', k=0.95, save_path='pca/'):
    """
    pyspark PCA wrapper to run a PCA that keeps components
    that account for a fixed percentage of total variance
    """
    n_features = len(df.first()[inputCol])
    base_pca = pyspark.ml.feature.PCA(k=n_features, inputCol=inputCol, outputCol=outputCol)
    base_model = base_pca.fit(df)
    k = base_model.explainedVariance.toArray().cumsum().searchsorted(k)
    pca = pyspark.ml.feature.PCA(k=k, inputCol=inputCol, outputCol=outputCol)
    model = pca.fit(df)
    pca.write().overwrite().save(os.path.join(save_path, 'pca'))
    print('Saved PCA!')
    model.write().overwrite().save(os.path.join(save_path, 'model'))
    print('Saved Model!')
    return model


def AutoKMeans(df, featuresCol, predictionCol, k_list, save_path):
    """
    pyspark KMeans wrapper to run several KMeans models
    """
    for k in k_list:
        print('K:', k)
        kmeans = pyspark.ml.clustering.KMeans(
            k=k,
            featuresCol=featuresCol,
            predictionCol=predictionCol
        )
        model = kmeans.fit(df)
        kmeans.write().overwrite().save(os.path.join(save_path, 'kmeans', f'k{k}'))
        model.write().overwrite().save(os.path.join(save_path, 'models', f'k{k}'))


def train_mode(df, **kwargs):
    # scale features for PCA
    scaler = pyspark.ml.feature.StandardScaler(
        inputCol=kwargs.get('inputCol', "embedding"),
        outputCol=kwargs.get('outputCol', "scaled_embedding"),
        withStd=kwargs.get('withStd', True),
        withMean=kwargs.get('withMean', True)
    )

    scalerModel = scaler.fit(df)
    scaler_save_path = kwargs.get(
        'scaler_save_path',
        's3://bucket-name/artifacts/scaler/scaler'
    )
    scaler.write().overwrite().save(scaler_save_path)

    scalerModel.write().overwrite().save(
        's3://bucket-name/artifacts/scaler/scalerModel'
    )

    # apply scaling to DataFrame
    df = scalerModel.transform(df).checkpoint()

    # build PCA model
    model = AutoPCA(
        df.sample(False, kwargs.get('train_prop', 0.25)),
        save_path=kwargs.get('pca_save_path', 's3://bucket-name/artifacts/pca'))

    # apply PCA to DataFrame
    df = model.transform(df)

    save_path = kwargs.get(
        'kmeans_save_path',
        's3://bucket-name/artifacts/clustering'
    )
    k_list = kwargs.get('k_list', [20, 40, 60, 80, 100, 150, 200, 300, 400, 500])

    AutoKMeans(
        df.sample(False, kwargs.get('train_prop', 0.25)),
        featuresCol='pcs',
        predictionCol='cluster',
        k_list=k_list,
        save_path=save_path
    )

    # check the SSE for each of the KMeans - elbow point was at k=150
    s3_urls = [os.path.join(save_path, 'models', 'k' + str(k)) for k in k_list]
    for u in s3_urls:
        model = pyspark.ml.clustering.KMeansModel.load(u)
        print(f'File: {u} | SSE: {model.computeCost(df)}')

    # load model of k=150
    model = pyspark.ml.clustering.KMeansModel.load(
        's3://bucket-name/artifacts/clustering/models/k150'
    )

    # apply KMeans clustering to DataFrame
    df = model.transform(df)

    return df


def predict_mode(df, **kwargs):
    scaler_path = kwargs.get(
        'scaler_path',
        's3://bucket-name/artifacts/scaler/scalerModel'
    )
    pca_path = kwargs.get(
        'pca_path',
        's3://bucket-name/artifacts/pca/model'
    )
    kmeans_path = kwargs.get(
        'kmeans_path',
        's3://bucket-name/artifacts/clustering/models/k150'
    )

    scaler = pyspark.ml.feature.StandardScalerModel.load(scaler_path)
    pca = pyspark.ml.feature.PCAModel.load(pca_path)
    kmeans = pyspark.ml.clustering.KMeansModel.load(kmeans_path)

    df = scaler.transform(df)
    df = pca.transform(df)
    df = kmeans.transform(df)

    return df


def main(path, mode):
    # setup SparkSession
    spark = SparkSession.builder.config("spark.executor.memory", "16g").getOrCreate()
    spark.sparkContext.setCheckpointDir('hdfs:///ckpt')
    sc.install_pypi_package('pandas')

    # read data
    schema = StructType([
        StructField('tweet_id', StringType()),
        StructField('embedding', StringType())
    ])
    df = spark.read.csv(path, sep=' ', schema=schema).repartition(1000)

    # transform embeddings from string to VectorUDT
    df = df.select('tweet_id', f.split('embedding', ',').cast('array<float>').alias('embedding'))
    extended_df = df.select('tweet_id', *[df.embedding[i] for i in range(768)])
    extended_df = extended_df.select(
        'tweet_id',
        f.array(['embedding[' + str(i) + ']' for i in range(768)]).alias('embedding')
    )
    array_to_vector_udf = f.udf(lambda a: Vectors.dense(a), VectorUDT())

    extended_df = extended_df.select(
        'tweet_id',
        array_to_vector_udf('embedding').alias('embedding')
    ).checkpoint()

    if mode == 'train':
        df = train_mode(df)
    else:
        df = predict_mode(df)

    # partition and save DataFrame
    df = df.withColumn('tweet_hash', f.substring('tweet_id', 1, 2))
    df.write.partitionBy('tweet_hash').parquet(
        's3://bucket-name/data/textEncodings/tweets_extended'
    )


def topic_history(raw_df, tweet_df):
    """
    This function deals with the creation of the past topic-interaction features of the users,
    the idea is that we want the model to know the past reactions of the users to different tweet
    topics. For example: we want the model to know that user 1234 liked 80% and retweeted 12% of
    previous tweets of the topic politics, while only liked 1% of sports. If the model also identifies
    that the prediction tweet is of topic politics, it is much more likely that the user will like it.
    """

    # load and partition raw data
    df = parse_data(raw_df, has_labels=True)
    df = df.withColumn('tweet_hash', F.substring('tweet_id', 1, 2))
    df = df.repartition('tweet_hash')

    #load the extended tweet data, the one that includes the KMeans cluster detected for each tweet
    twt_df = spark.read.parquet(tweet_df)

    #join training DataFrame with the tweet features
    df = df.join(twt_df, 'tweet_id', 'inner').select(
        'engaging_user_id', 'cluster',
        'reply_timestamp', 'retweet_timestamp',
        'retweet_with_comment_timestamp', 'like_timestamp'
    )

    #transform indicator features from timestamp to binary
    df = df.withColumn(
        'indicator_reply',
        F.when(F.col('reply_timestamp').isNotNull(), 1).otherwise(0)
    )
    df = df.withColumn(
        'indicator_retweet',
        F.when(F.col('retweet_timestamp').isNotNull(), 1).otherwise(0)
    )
    df = df.withColumn(
        'indicator_retweet_with_comment',
        F.when(F.col('retweet_with_comment_timestamp').isNotNull(), 1).otherwise(0)
    )
    df = df.withColumn(
        'indicator_like',
        F.when(F.col('like_timestamp').isNotNull(), 1).otherwise(0)
    )
    df = df.select(
        'engaging_user_id', 'cluster', 'indicator_reply',
        'indicator_retweet', 'indicator_retweet_with_comment', 'indicator_like'
    )
    df = df.checkpoint()

    # pivot the DataFrame to create features for each user/topic combination
    # we want to capture the number of times the user was exposed to tweets of that topic, and what
    # proportion of those exposures resulted in positive results for each engagement type
    df = df.groupBy('engaging_user_id') \
           .pivot('cluster', values=[c for c in range(150)]) \
           .agg(F.count(F.col('indicator_reply')).alias('reply_count_num'),
                F.avg(F.col('indicator_reply')).alias('reply_prop_num'),
                F.count(F.col('indicator_retweet')).alias('retweet_count_num'),
                F.avg(F.col('indicator_retweet')).alias('retweet_prop_num'),
                F.count(F.col('indicator_retweet_with_comment')).alias('retweet_w_comment_count_num'),
                F.avg(F.col('indicator_retweet_with_comment')).alias('retweet_w_comment_prop_num'),
                F.count(F.col('indicator_like')).alias('like_count_num'),
                F.avg(F.col('indicator_like')).alias('like_prop_num'))

    #impute by zeroes. In this case there's no problem doing so because this imputation says:
    #if he was never exposed to the topic, then the amount of exposures is zero and by consequence
    #the proportion of positive engagements is also zero.
    df = df.na.fill(0)

    #partition and write the DataFrame
    df = df.withColumn('engaging_user_hash', F.substring('engaging_user_id', 1, 2))
    df.write.partitionBy('engaging_user_hash').parquet('s3://bucket-name/data/textEncodings/user_topics', mode='overwrite')


if __name__ == '__main__':
    train = True
    path = 's3://bucket-name/data/textEncodings/embs.txt'

    mode = 'train' if train else 'predict'
    main(path, mode)
    topic_history('s3://bucket-name/data/raw/training.tsv',
                  's3://bucket-name/data/textEncodings/tweets_extended')
