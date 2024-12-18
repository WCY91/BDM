from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, col, to_date, lit
from pyspark.sql.types import DateType
from datetime import datetime

input_file = "spacenews.csv"

spark = SparkSession.builder.appName("hw1_task1").getOrCreate()
df = spark.read.csv(input_file, sep=',', header = True, inferSchema = True)
spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")

#drop the row if missing data in feilds "Title" or "Date"
df = df.na.drop(subset=["title", "date"])

#split the feild "Title" into words and explode into rows
words = df.select(explode(split(col("title"), " ")).alias("word"))

#group by words and count their frequencies
word_count = words.groupBy("word").count()

#order by the count in descending order
sorted_words = word_count.orderBy(col("count").desc())

sorted_words.show(truncate=False)
#sorted_words.coalesce(1).write.format("csv").option("header", "true").mode("overwrite").save("hw1-1_output")



#convert the feild "Date" to date type if it is not already and drop the NULL row
df = df.withColumn("date", to_date(col("date"), "MMM dd, yyyy"))
df_clean = df.filter(col("date").isNotNull())

words_day = df_clean.select(col("date"), explode(split(col("title"), " ")).alias("word"))

count_day = words_day.groupBy("date", "word").count()

sorted_day = count_day.orderBy(col("date"), col("count").desc())

sorted_day.show(truncate=False)


#sorted_day.coalesce(1).write.format("csv").option("header", "true").mode("overwrite").save("hw1-1_output")


sorted_words = sorted_words.withColumn("date", lit(None).cast(DateType()))
merge_df = sorted_day.unionByName(sorted_words)
#test 
#merge_df.show(truncate=False)

merge_df.coalesce(1).write.format("csv").option("header", "true").mode("overwrite").save("hw1-1_output")