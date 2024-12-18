import os 
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, DateType, StringType
from pyspark.sql.functions import explode, split, lower, count, to_date
from pyspark.sql import functions as F
from pyspark.sql.functions import avg

spark = SparkSession.builder.appName("HW4_1").getOrCreate()

movies_df = spark.read.csv('movies.dat', sep='::', inferSchema=True)
movies_df = movies_df.select(col("_c0").alias("movie_id"), col("_c1").alias("movie_name"))
movies_df = movies_df.na.drop()
movies_df = movies_df.select("movie_id","movie_name")

ratings_df = spark.read.csv('ratings.dat', sep='::', inferSchema=True)
ratings_df = ratings_df.select(col("_c1").alias("movie_id"), col("_c2").alias("score"))
ratings_df = ratings_df.na.drop()
ratings_movie_df = ratings_df.groupBy("movie_id").agg(avg("score").alias("avg_rating"))

movies_rating_rank_df = movies_df.join(ratings_movie_df, "movie_id")
movies_rating_rank_df = movies_rating_rank_df.orderBy("avg_rating", ascending=False)

movies_rating_rank_df = movies_rating_rank_df.select("movie_name", "avg_rating").orderBy("avg_rating", ascending=False)
movies_rating_rank_df.show()

file_name = "hw(1)_movie_avg_ratings_rank.txt"
movies_rating_rank_df.coalesce(1).write.format("txt").option("header", "true").mode("overwrite").save(file_name)
