from pyspark.sql import SparkSession
from pyspark.sql.functions import lead, col, monotonically_increasing_id, posexplode, split, sha2, input_file_name, trim, rand
from pyspark.sql.window import Window

def create_paragraphs_df(spark, input_dir):
    """
    Loads full articles and splits each article's content into actual paragraphs.
    Returns doc_id, paragraph_index, and individual paragraph content.
    """
    print("Loading JSON input")
    df = spark.read.option("multiLine", True).json(f"{input_dir}/**/*.json")

    # Split article text into paragraphs (assumes \\n\\n separates them)
    df = df.select(
        sha2(input_file_name(), 256).alias("doc_id"),
        split(col("content"), r"\\n\\n").alias("paragraphs")
    )

    # Explode paragraphs into rows and index them
    df = df.select("doc_id", posexplode("paragraphs").alias("paragraph_index", "content"))
    return df.filter(trim(col("content")) != "")

def create_triplets(df):
    """
    Creates triplet training examples from paragraph DataFrame.
    Anchor and positive are adjacent paragraphs from the same doc.
    Negatives are random paragraphs from other docs.
    """

    w = Window.partitionBy("doc_id").orderBy("paragraph_index")

    anchor_positive_df = df.withColumn("positive", lead("content", 1).over(w)) \
                           .select("doc_id", "content", "positive") \
                           .withColumnRenamed("content", "anchor") \
                           .filter(col("positive").isNotNull())

    # Random negatives from other docs
    negatives_df = df.select(col("doc_id"), col("content").alias("negative")).orderBy(rand())

    # Add matching indexes
    anchor_positive_df = anchor_positive_df.withColumn("pair_index", monotonically_increasing_id())
    negatives_df = negatives_df.withColumn("neg_index", monotonically_increasing_id())

    # Alias the dataframes to avoid ambiguity
    ap = anchor_positive_df.alias("ap")
    neg = negatives_df.alias("neg")

    # Join and disambiguate using qualified column names
    triplets = ap.join(
        neg,
        col("ap.pair_index") == col("neg.neg_index")
    ).filter(col("ap.doc_id") != col("neg.doc_id")) \
     .select(col("ap.anchor"), col("ap.positive"), col("neg.negative"))

    return triplets


if __name__ == "__main__":
    INPUT_DIR = r"../data/processed/wikidata_json"
    OUTPUT_DIR = r"../data/processed/triplets/combined"

    spark = SparkSession.builder \
        .appName("TripletCreator") \
        .master("local[*]") \
        .config("spark.driver.memory", "24g") \
        .config("spark.sql.shuffle.partitions", "100") \
        .config("spark.local.dir", "../spark-temp") \
        .config("spark.driver.maxResultSize", "2g") \
        .getOrCreate()

    df = create_paragraphs_df(spark, INPUT_DIR)
    triplets = create_triplets(df)

    print(f"Writing triplets to Spark part files: {OUTPUT_DIR}")
    triplets.coalesce(1).write.mode("overwrite").json(OUTPUT_DIR)

    spark.stop()
