import os 
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, DateType, StringType
from pyspark.sql.functions import explode, split, lower, count, to_date
from pyspark.sql import functions as F
from pyspark.sql.functions import avg

spark = SparkSession.builder.appName("HW4_1").getOrCreate()

movies_df = spark.read.csv('movies.dat', sep='::', inferSchema=True)
movies_df = movies_df.select(col("_c0").alias("movie_id"), col("_c1").alias("movie_name"), col("_c2").alias("genres"))
movies_df = movies_df.na.drop()
ratings_df = spark.read.csv('ratings.dat', sep='::', inferSchema=True)
ratings_df = ratings_df.select(col("_c0").alias("user_id") ,col("_c1").alias("movie_id"), col("_c2").alias("score"),col("_c3").alias("timestamp"))
ratings_df = ratings_df.na.drop()
users_df = spark.read.csv('users.dat',sep="::",inferSchema = True)
users_df = users_df.select(col("_c0").alias("user_id"),col("_c1").alias("gender"),col("_c2").alias("age"),col("_c3").alias("occupation"),col("_c4").alias("zip"))
users_df = users_df.na.drop()

movie_rating_movie_df = movies_df.join(ratings_df, "movie_id")
user_movie_rating_df = movie_rating_movie_df.join(users_df, "user_id")

user_avg_rating_df = user_movie_rating_df.groupBy("user_id").agg(avg("score").alias("avg_score"))
user_avg_rating_df = user_avg_rating_df.select("user_id", "avg_score").orderBy("avg_score", ascending=False)
user_avg_rating_df.show()
file_name = "hw(3)_user_avg_rating.txt"
user_avg_rating_df.coalesce(1).write.format("txt").option("header", "true").mode("overwrite").save(file_name)

user_genre_avg_rating_df = user_movie_rating_df.groupBy("user_id", "genres").agg(avg("score").alias("avg_score"))
user_genre_avg_rating_df = user_genre_avg_rating_df.select("user_id", "genres", "avg_score").orderBy("avg_score", ascending=False)
user_genre_avg_rating_df.show()
file_name = "hw(3)_user_genre_avg_rating.txt"
user_genre_avg_rating_df.coalesce(1).write.format("txt").option("header", "true").mode("overwrite").save(file_name)
