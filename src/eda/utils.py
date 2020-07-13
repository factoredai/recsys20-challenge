from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (StructType, StructField, StringType, LongType,
                               IntegerType, BooleanType)


def build_schema(has_labels):
    """Builds the schema for processing the Twitter challenge data as PySpark DataFrames.

    Arguments:
        has_labels {bool} -- True if the data includes labels

    Returns:
        pyspark.sql.types.StructType -- schema to build the DataFrame with.
    """
    if has_labels:
        schema = StructType(
            [
                StructField('text_tokens', StringType()),
                StructField('hashtags', StringType()),
                StructField('tweet_id', StringType()),
                StructField('present_media', StringType()),
                StructField('present_links', StringType()),
                StructField('present_domains', StringType()),
                StructField('tweet_type', StringType()),
                StructField('language', StringType()),
                StructField('tweet_timestamp', LongType()),
                StructField('engaged_with_user_id', StringType()),
                StructField('engaged_with_user_follower_count', IntegerType()),
                StructField('engaged_with_user_following_count', IntegerType()),
                StructField('engaged_with_user_is_verified', BooleanType()),
                StructField('engaged_with_user_account_creation', LongType()),
                StructField('engaging_user_id', StringType()),
                StructField('engaging_user_follower_count', IntegerType()),
                StructField('engaging_user_following_count', IntegerType()),
                StructField('engaging_user_is_verified', BooleanType()),
                StructField('engaging_user_account_creation', LongType()),
                StructField('engagee_follows_engager', BooleanType()),
                StructField('reply_timestamp', LongType()),
                StructField('retweet_timestamp', LongType()),
                StructField('retweet_with_comment_timestamp', LongType()),
                StructField('like_timestamp', LongType())
            ]
        )
    else:
        schema = StructType(
            [
                StructField('text_tokens', StringType()),
                StructField('hashtags', StringType()),
                StructField('tweet_id', StringType()),
                StructField('present_media', StringType()),
                StructField('present_links', StringType()),
                StructField('present_domains', StringType()),
                StructField('tweet_type', StringType()),
                StructField('language', StringType()),
                StructField('tweet_timestamp', LongType()),
                StructField('engaged_with_user_id', StringType()),
                StructField('engaged_with_user_follower_count', IntegerType()),
                StructField('engaged_with_user_following_count', IntegerType()),
                StructField('engaged_with_user_is_verified', BooleanType()),
                StructField('engaged_with_user_account_creation', LongType()),
                StructField('engaging_user_id', StringType()),
                StructField('engaging_user_follower_count', IntegerType()),
                StructField('engaging_user_following_count', IntegerType()),
                StructField('engaging_user_is_verified', BooleanType()),
                StructField('engaging_user_account_creation', LongType()),
                StructField('engagee_follows_engager', BooleanType())
            ]
        )
    return schema


def parse_data(path, has_labels, schema='auto'):
    """Parses the training data for the Twitter RecSys Challenge.

    Arguments:
        path {str} -- path of the raw data
        has_labels {bool} -- True if the data includes labels

    Keyword Arguments:
        schema {str} -- schema to use when reading.
                        If 'auto' uses a predefined schema (default: {'auto'})

    Returns:
        pyspark.sql.DataFrame -- DataFrame with the parsed data
    """
    spark = SparkSession.builder.appName("twitter").getOrCreate()
    if schema == 'auto':
        schema = build_schema(has_labels)
    df = spark.read.csv(path, schema=schema, sep='\x01', encoding='utf-8',
                        ignoreLeadingWhiteSpace=True, ignoreTrailingWhiteSpace=True)
    df = df.withColumn('text_tokens', F.split('text_tokens', '\t'))
    df = df.withColumn('hashtags', F.split('hashtags', '\t'))
    df = df.withColumn('present_media', F.split('present_media', '\t'))
    df = df.withColumn('present_links', F.split('present_links', '\t'))
    df = df.withColumn('present_domains', F.split('present_domains', '\t'))
    return df


def get_intersection_stats(df1, df2, on):
    """Gets some statistics about the intersection between two DataFrames, used
    to analyze how many records are shared between them.

    Arguments:
        df1 {pyspark.sql.DataFrame} -- DataFrame 1
        df2 {pyspark.sql.DataFrame} -- DataFrame 2
        on {str} -- name of the column to join on

    Returns:
        (int, float) -- returns the amount of rows in common and the proportion of rows in common.
    """
    unique_df1 = df1.select(on).distinct()
    unique_df2 = df2.select(on).distinct()
    int_df = unique_df1.join(unique_df2, on, "inner")
    int_df_count = int_df.count()
    return (int_df_count, int_df_count/unique_df1.count())
