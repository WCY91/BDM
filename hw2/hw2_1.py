
import os
memory = '4g'
pyspark_submit_args = ' --driver-memory '+memory+' pyspark-shell'
os.environ['PYSPARK_SUBMIT_ARGS'] = pyspark_submit_args


from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DateType, StringType
from pyspark.sql.functions import explode, split, lower, count, to_date
from pyspark.sql import functions as F


spark = SparkSession.builder.appName("HW2_1").getOrCreate()

df = spark.read.csv("News_Final.csv", sep=',', header=True, inferSchema=True)
df = df.na.drop(subset=["Title","Headline","PublishDate","Topic"])  

df_selected = df.select("Title","Headline","PublishDate","Topic")
df_title_words = df_selected.withColumn("TitleWords", explode(split(lower(F.col("Title")), " "))) #分詞
df_headline_words = df_selected.withColumn("HeadlineWords", explode(split(lower(F.col("Headline")), " ")))#分詞

df_title_word_count = df_title_words.groupBy("TitleWords").count().orderBy("count", ascending=False)
df_headline_word_count = df_headline_words.groupBy("HeadlineWords").count().orderBy("count", ascending=False)

# df_title_word_count.show()
# df_headline_word_count.show()

df_title_word_count.coalesce(1).write.format("csv").option("header", "true").mode("overwrite").save("hw2-1_title_count_output")
df_headline_word_count.coalesce(1).write.format("csv").option("header", "true").mode("overwrite").save("hw2-1_headline_count_output")

df_title_word_datecount = df_title_words.groupBy("PublishDate", "TitleWords").agg(F.count("TitleWords").alias("count")).orderBy("PublishDate", "count", ascending=[True, False])
df_headline_word_datecount = df_headline_words.groupBy("PublishDate", "HeadlineWords").agg(F.count("HeadlineWords").alias("count")).orderBy("PublishDate", "count", ascending=[True, False])

# df_headline_word_datecount.show()
# df_title_word_datecount.show()

df_title_word_datecount.coalesce(1).write.format("csv").option("header", "true").mode("overwrite").save("hw2-1_title_date_count_output")
df_headline_word_datecount.coalesce(1).write.format("csv").option("header", "true").mode("overwrite").save("hw2-1_headline_date_count_output")

df_title_word_topiccount = df_title_words.groupBy("Topic", "TitleWords").agg(F.count("TitleWords").alias("count")).orderBy("count", ascending=False)
df_headline_word_topiccount = df_headline_words.groupBy("Topic", "HeadlineWords").agg(F.count("HeadlineWords").alias("count")).orderBy("count", ascending=False)

df_title_word_topiccount.coalesce(1).write.format("csv").option("header", "true").mode("overwrite").save("hw2-1_title_topic_count_output")
df_headline_word_topiccount.coalesce(1).write.format("csv").option("header", "true").mode("overwrite").save("hw2-1_headline_topic_count_output")


