from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, lead, col, monotonically_increasing_id, posexplode, split, sha2, input_file_name, trim, rand
from pyspark.sql.window import Window
import os
import json


def create_article_metadata(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.isfile(output_path):
        print("Metadata already exists.")
        return

    print("creating metadata")
    # Add filename as a column (source file)
    df = df.withColumn("source_file", input_file_name())

    # Select desired metadata fields
    metadata_df = df.select(
        col("title").alias("title"),
        col("url").alias("url"),
        col("source_file")
    ).fillna({"title": "Untitled", "url": ""})

    # Collect to driver and convert to Python list of dicts
    # print("converting to df")
    # article_metadata = metadata_df.toPandas().to_dict(orient="records")

    # Save to JSON
    # print(f"Saving {len(article_metadata)} metadata entries to {output_path}")
    # with open(output_path, "w", encoding="utf-8") as f:
    #     json.dump(article_metadata, f, ensure_ascii=False, indent=2)
    metadata_df.write.mode("overwrite").json(output_path)
    # print(f"Saved {len(article_metadata)} metadata entries to {output_path}")


def create_paragraphs_df(df):
    """
    Loads full articles and splits each article's content into actual paragraphs.
    Returns doc_id, paragraph_index, and individual paragraph content.
    """
    # Split article text into paragraphs (assumes \\n\\n separates them)
    df = df.select(
        sha2(input_file_name(), 256).alias("doc_id"),
        split(col("content"), r"\\n\\n").alias("paragraphs")
    )

    # Explode paragraphs into rows and index them
    df = df.select("doc_id", posexplode("paragraphs").alias("paragraph_index", "content"))
    return df.filter(trim(col("content")) != "")


def create_paragraph_metadata(df, output_path):
    os.makedirs(output_path, exist_ok=True)

    # Add source_file column if needed
    df = df.withColumn("source_file", input_file_name())

    # Select relevant fields
    metadata_df = df.select(
        col("content").alias("text"),
        col("doc_id"),
        col("source_file")
    )

    print(f"Saving paragraph metadata to: {output_path}")
    metadata_df.write.mode("overwrite").json(output_path)
    print("✅ Paragraph metadata written as JSONL (Spark JSON format)")


def create_random_triplets(df):
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
    OUTPUT_DIR = r"../data/processed/triplets/parts"
    ARTICLE_METADATA_OUTPUT_PATH = r"../data/custom_model/article_metadata.json"
    PARAGRAPH_METADATA_OUTPUT_DIR = r"../data/custom_model/paragraph_metadata"

    spark = SparkSession.builder \
        .appName("TripletCreator") \
        .master("local[*]") \
        .config("spark.driver.memory", "20g") \
        .config("spark.sql.shuffle.partitions", "100") \
        .config("spark.local.dir", "../spark-temp") \
        .config("spark.driver.maxResultSize", "2g") \
        .getOrCreate()

    print("Loading JSON input")
    input_df = spark.read.option("multiLine", True).json(f"{INPUT_DIR}/**/*.json")
    create_article_metadata(input_df, ARTICLE_METADATA_OUTPUT_PATH)

    df = create_paragraphs_df(input_df)
    create_paragraph_metadata(df, PARAGRAPH_METADATA_OUTPUT_DIR)

    triplets = create_random_triplets(df)
    print(f"Writing triplets to Spark part files: {OUTPUT_DIR}")
    triplets.write.mode("overwrite").json(OUTPUT_DIR)

    total_lines = spark.read.json(f"{INPUT_DIR}/*.json", multiLine=False).count()
    print("Total lines across training files:", total_lines)

    spark.stop()
