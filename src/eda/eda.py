"""
This EDA allowed us to extract some interesting takeaways,
which would be helpful in future modeling:

    * Over 70% of tweets only appear once in the training set.
    * There is one specific tweet that appears over 30k times in the training set.
    * 50% of tweet authors appear at least twice in the training set, with one user accounting
      for over 400k interactions.
    * 60% of engagers appear at least 4 times in the training set.
    * There are 60M unique tweets in the training dataset.
    * There are no tweets in common between training and submission datasets.
    * 80% of users (posters and engagers) in submission are also present
      in training (either as posters or engagers)
    * 80% of tweet authors in submission are also present in training.
    * 73% of engaging users in submission are also present in training.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from utils import parse_data, get_intersection_stats

spark = SparkSession.builder \
                    .appName("twitter") \
                    .config("spark.executor.memory", "16g") \
                    .config("spark.driver.memory", "16g") \
                    .config("spark.driver.maxResultSize", "8g") \
                    .config("spark.memory.offHeap.enabled", "true") \
                    .config("spark.memory.offHeap.size", "16g") \
                    .getOrCreate()

train_df = parse_data(
    's3://bucket-name/data/raw/training.tsv',
    has_labels=True).repartition(5000)
submission_df = parse_data(
    's3://bucket-name/data/raw/evaluation.tsv',
    has_labels=False).repartition(100)

# create a list with percentiles for analyzing the count distributions of some variables
stats = [
    "count", "min", "10%", "20%", "30%", "40%", "50%",
    "60%", "70%", "80%", "90%", "95%", "99%", "max"
]

# appearance of tweets
train_df.groupBy("tweet_id").count().select("count").summary(*stats).show()

# appearance of engaged_with_user
train_df.groupBy("engaged_with_user_id").count().select("count").summary(*stats).show()

# appearance of engaging_user
train_df.groupBy("engaging_user_id").count().select("count").summary(*stats).show()

# common tweets between submission and training
get_intersection_stats(submission_df, train_df, on="tweet_id")

# common engaged_with_user between submission and training
get_intersection_stats(submission_df, train_df, on="engaged_with_user_id")

# common engaging_user between submission and training
get_intersection_stats(submission_df, train_df, on="engaging_user_id")

# common users between submission and training
s_ids = (
    submission_df.withColumn(
        "ids",
        F.array("engaged_with_user_id", "engaging_user_id")).select("ids")) \
        .withColumn("ids", F.explode("ids")).distinct()
t_ids = (
    train_df.withColumn(
        "ids",
        F.array("engaged_with_user_id", "engaging_user_id")).select("ids")) \
        .withColumn("ids", F.explode("ids")).distinct()
get_intersection_stats(s_ids, t_ids, on="ids")
