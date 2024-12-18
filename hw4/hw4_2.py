import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg

spark = SparkSession.builder.appName("HW4_1").getOrCreate()

movies_df = spark.read.csv('movies.dat', sep='::', inferSchema=True)
movies_df = movies_df.select(col("_c0").alias("movie_id"), col("_c1").alias("movie_name"), col("_c2").alias("genres"))
movies_df = movies_df.na.drop()
ratings_df = spark.read.csv('ratings.dat', sep='::', inferSchema=True)
ratings_df = ratings_df.select(col("_c0").alias("user_id"), col("_c1").alias("movie_id"), col("_c2").alias("score"), col("_c3").alias("timestamp"))
ratings_df = ratings_df.na.drop()
users_df = spark.read.csv('users.dat', sep="::", inferSchema=True)
users_df = users_df.select(col("_c0").alias("user_id"), col("_c1").alias("gender"), col("_c2").alias("age"), col("_c3").alias("occupation"), col("_c4").alias("zip"))
users_df = users_df.na.drop()

movie_rating_movie_df = movies_df.join(ratings_df, "movie_id")
user_movie_rating_df = movie_rating_movie_df.join(users_df, "user_id")

user_movie_rating_gender_rank = user_movie_rating_df.groupBy("gender", "movie_name").agg(avg("score").alias("avg_score"))
user_movie_rating_gender_rank = user_movie_rating_gender_rank.select("movie_name", "gender", "avg_score").orderBy("avg_score", ascending=False)
user_movie_rating_gender_rank.show()
file_name = "hw(2)_user_movie_rating_gender_rank.txt"
user_movie_rating_gender_rank.coalesce(1).write.format("txt").option("header", "true").mode("overwrite").save(file_name)

user_movie_rating_age_rank = user_movie_rating_df.groupBy("age", "movie_name").agg(avg("score").alias("avg_score"))
user_movie_rating_age_rank = user_movie_rating_age_rank.select("movie_name", "age", "avg_score").orderBy("avg_score", ascending=False)
user_movie_rating_age_rank.show()
file_name = "hw(2)_user_movie_rating_age_rank.txt"
user_movie_rating_age_rank.coalesce(1).write.format("txt").option("header", "true").mode("overwrite").save(file_name)

user_movie_rating_occupation_rank = user_movie_rating_df.groupBy("occupation", "movie_name").agg(avg("score").alias("avg_score"))
user_movie_rating_occupation_rank = user_movie_rating_occupation_rank.select("movie_name", "occupation", "avg_score").orderBy("avg_score", ascending=False)
user_movie_rating_occupation_rank.show()
file_name = "hw(2)_user_movie_rating_occupation_rank.txt"
user_movie_rating_occupation_rank.coalesce(1).write.format("txt").option("header", "true").mode("overwrite").save(file_name)
