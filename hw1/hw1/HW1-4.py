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

spark = SparkSession.builder.appName("HW1_4").getOrCreate()

df = spark.read.csv("spacenews.csv", sep=',', header=True, schema=schema)
df = df.na.drop(subset=["date"])  
spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
df = df.withColumn("date", to_date("date", "MMM dd, yyyy"))

df = df.filter(F.col("date").isNotNull())

df_formated = df.withColumn("title_lower",lower(F.col("title")))
df_formated = df_formated.withColumn("postexcerpt_lower",lower(F.col("postexcerpt")))

df_filtered = df_formated.filter(F.col("postexcerpt_lower").contains("space") & F.col("title_lower").contains("space"))

df_filtered.coalesce(1).write.format("csv").option("header", "true").mode("overwrite").save("hw1-4_output")
df_filtered.select(["date","title","postexcerpt"]).show()



