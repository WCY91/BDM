from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split, lower, count, udf, array_intersect, array, lit, size
from pyspark.sql.types import StringType
import pyspark.sql.functions as f
import sys
sys.stdout.reconfigure(encoding='utf-8')
# Initialize SparkSession
spark = SparkSession.builder.appName("OptimizedCooccurrenceMatrix").getOrCreate()

# Read data
path = "News_Final.csv"  # Replace with your actual path
df = spark.read.option("escape", '"').csv(path, header=True)

# Fill missing values
df = df.na.fill("missing")

# Stopwords list
stopwords = set(["the", "and", "to", "of", "in", "a", "is", "for", "on", "that", "with", "as", "at", "by"])

# Define a function to clean and lower-case strings
def lower_clean_str(x):
    if x is None:
        return ""
    punc = '!"#”$%&\'()*+—–,./:;<=>?@[\\]^_’‘`{|}~-…'
    lowercased_str = x.lower()
    for ch in punc:
        lowercased_str = lowercased_str.replace(ch, ' ')
    return lowercased_str

lower_clean_str_udf = udf(lower_clean_str, StringType())

# Clean and lower-case text
df = df.withColumn("Title", lower_clean_str_udf(col("Title")))
df = df.withColumn("Headline", lower_clean_str_udf(col("Headline")))

# Filter specific topics
topic_list = ['economy', 'microsoft', 'obama', 'palestine']
df = df.filter(df["Topic"].isin(topic_list))

# Function to calculate word frequency
def calculate_word_frequency(df, column_name, stopwords):
    word_df = df.withColumn("Word", explode(split(col(column_name), " "))) \
                .filter(~col("Word").isin(stopwords)) \
                .groupBy("Topic", "Word") \
                .count() \
                .orderBy("Topic", f.desc("count"))
    return word_df

title_word_count = calculate_word_frequency(df, "Title", stopwords)
headline_word_count = calculate_word_frequency(df, "Headline", stopwords)

# Get top 100 words per topic
def get_top_words(word_count_df, n=100):
    from pyspark.sql.window import Window
    window_spec = Window.partitionBy("Topic").orderBy(f.desc("count"))
    return word_count_df.withColumn("rank", f.row_number().over(window_spec)) \
                        .filter(col("rank") <= n) \
                        .drop("rank")

title_top100 = get_top_words(title_word_count)
headline_top100 = get_top_words(headline_word_count)

# Function to compute co-occurrence matrix
def compute_cooccurrence_matrix(df_sentences, df_top_words, topic, column_name):
    # Get the list of top 100 words for the topic
    top_words_list = df_top_words.filter(col("Topic") == topic).select("Word").rdd.flatMap(lambda x: x).collect()
    
    # Broadcast the top words list
    broadcast_top_words = spark.sparkContext.broadcast(top_words_list)
    
    # Filter sentences of the given topic
    df_topic_sentences = df_sentences.filter(col("Topic") == topic).select(column_name)
    
    # Split sentences into words
    df_words = df_topic_sentences.select(split(col(column_name), " ").alias("words"))
    
    # Filter words to keep only top words
    top_words_array = array([lit(x) for x in broadcast_top_words.value])
    df_filtered_words = df_words.select(array_intersect(col("words"), top_words_array).alias("words"))
    
    # Filter out empty word lists
    df_filtered_words = df_filtered_words.filter(size(col("words")) > 1)
    
    # Assign an ID to each sentence
    df_filtered_words = df_filtered_words.withColumn("id", f.monotonically_increasing_id())
    
    # Explode words and prepare for co-occurrence calculation
    df_exploded_words = df_filtered_words.select("id", explode("words").alias("word"))
    
    # Self-join to get word pairs
    df_word_pairs = df_exploded_words.alias("df1").join(
        df_exploded_words.alias("df2"),
        on="id"
    ).filter(col("df1.word") < col("df2.word"))
    
    # Count co-occurrences
    df_cooccurrence = df_word_pairs.groupBy("df1.word", "df2.word").count()
    
    # Pivot to create the co-occurrence matrix
    cooccurrence_matrix = df_cooccurrence.groupBy("df1.word") \
                                         .pivot("df2.word") \
                                         .agg(f.first("count")) \
                                         .fillna(0)
    return cooccurrence_matrix

# Generate co-occurrence matrices for each topic
for topic in topic_list:
    print(f"Matrix: {topic} Title")
    matrix_title = compute_cooccurrence_matrix(df, title_top100, topic, "Title")
    matrix_title.coalesce(1).write.format("csv").option("header", "true").mode("overwrite").save(f"{topic}matrix_title")

    print(f"Matrix: {topic} Headline")
    matrix_headline = compute_cooccurrence_matrix(df, headline_top100, topic, "Headline")
    matrix_headline.coalesce(1).write.format("csv").option("header", "true").mode("overwrite").save(f"{topic}matrix_title")


#import sys
#sys.stdout.reconfigure(encoding='utf-8')


#spark-submit --master spark://192.168.56.101:7077 --conf spark.driver.host=192.168.56.101 --conf spark.driver.memory=4g --conf spark.executor.memory=4g --conf spark.executor.cores=4 --conf spark.reducer.maxReqsInFlight=10 --conf spark.reducer.maxBlocksInFlightPerAddress=10 /home/cluster0/hw2_4.py

