from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, col, to_date, lit
from pyspark.sql.types import DateType
from datetime import datetime

input_file = "spacenews.csv"

spark = SparkSession.builder.appName("hw1_task3").getOrCreate()
df = spark.read.csv(input_file, sep=',', header = True, inferSchema = True)
spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")

df = df.na.drop(subset=["author", "date"])

df = df.withColumn("date", to_date(col("date"), "MMM dd, yyyy"))
df = df.filter(col("date").isNotNull())

daily_count = df.groupBy("date").count()

total_article = df.count()

daily_percent = daily_count.withColumn("percentage", ((col("count")/total_article) * 100))

daily_percent.show(truncate=False)

daily_author_count = df.groupBy("date", "author").count()

daily_total = df.groupBy("date").count().withColumnRenamed("count", "total_articles")

author_precent = daily_author_count.join(daily_total, "date")

author_percent = author_precent.withColumn("percentage", ((col("count")/col("total_articles")) * 100))

author_percent.show(truncate=False)

daily_percent = daily_percent.withColumn("author", lit(None))
daily_percent = daily_percent.withColumn("total_articles", lit(None))
daily_percent = daily_percent.select("date", lit(None).alias("author"), col("count"), lit(None).alias("total_articles"), col("percentage"))
merge_df = author_percent.unionByName(daily_percent.select("date", "author", "count", "total_articles", "percentage"))
merge_df = merge_df.select("date", "author", "count", "total_articles", "percentage")
merge_df.coalesce(1).write.format("csv").option("header", "true").mode("overwrite").save("hw1-3_output")