from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DateType, StringType
from pyspark.sql.functions import explode, split, lower, count, to_date
from pyspark.sql import functions as F

schema = StructType([
    StructField("title", StringType(), True),	
    StructField("url", StringType(), True),	
    StructField("content", StringType(), True),
    StructField("author", StringType(), True),
    StructField("date", StringType(), True),
    StructField("postexcerpt", StringType(), True),
])

spark = SparkSession.builder.appName("HW1_2").getOrCreate()

df = spark.read.csv("spacenews.csv", sep=',', header=True, schema=schema)
df = df.na.drop(subset=["date"])  
spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
df = df.withColumn("date", to_date("date", "MMM dd, yyyy"))

df = df.filter(F.col("date").isNotNull())

df_selected = df.select("content", "date")
df.unpersist()
df_words = df_selected.withColumn("word", explode(split(lower(F.col("content")), " "))) #explode會多出很多word 因為explode用意就是產生類似key value
df_selected.unpersist()
df_words_1 = df_words.groupBy("word").count()
df_words_1 = df_words_1.orderBy(df_words["count"].desc())
df_words_1.coalesce(1).write.format("csv").option("header", "true").mode("overwrite").save("hw1-2_1output")

df_words_1.unpersist()
df_word_count = df_words.groupBy("date", "word").agg(F.count("word").alias("count")) #將explode產生的word透過groupBy去做統整
columns = ["date", "word", "count"]

df_word_count = df_word_count.select(columns).orderBy("date", "count", ascending=[True, False])
df_word_count = df_word_count.select([F.col(c).cast("string") for c in df_word_count.columns])
df_word_count.coalesce(1).write.format("csv").option("header", "true").mode("overwrite").save("hw1-2_2output")
df_word_count.show()



