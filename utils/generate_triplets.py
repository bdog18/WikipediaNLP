from pyspark.sql import SparkSession
from pyspark.sql.functions import col, row_number, rand, lit, monotonically_increasing_id
from tqdm import tqdm
import os
import glob

# Initialize Spark session
spark = SparkSession.builder \
    .appName("TripletCreator") \
    .master("local[*]") \
    .config("spark.driver.memory", "24g") \
    .config("spark.sql.shuffle.partitions", "100") \
    .config("spark.local.dir", "../spark-temp") \
    .getOrCreate()

# File paths
INPUT_DIR = "../data/processed/wikidata_json"
OUTPUT_DIR = "../data/triplets/parts"

print("[1/6] Loading JSON input")
json_dirs = glob.glob(f"{INPUT_DIR}/*")

df = None
for path in tqdm(json_dirs, desc="Processing folders", unit="dir"):
    json_files = glob.glob(os.path.join(path, "*.json"))
    for json_file in json_files:
        try:
            current = spark.read.option("multiLine", True).json(json_file)
            if df is None:
                df = current
                tqdm.write(f"→ Initialized base DataFrame from {json_file}")
            else:
                df = df.unionByName(current)
        except Exception as e:
            tqdm.write(f"    Skipping {json_file} due to error: {e}")

# dataframes = []
# for path in tqdm(json_dirs, desc="Processing folders", unit="dir"):
#     json_files = glob.glob(os.path.join(path, "*.json"))
#     for json_file in json_files:
#         try:
#             df = spark.read.option("multiLine", True).json(json_file)
#             dataframes.append(df)
#         except Exception as e:
#             tqdm.write(f"    Skipping {json_file} due to error: {e}")

# df = dataframes[0]
# print(f"→ Starting union with base DataFrame (1 of {len(dataframes)})")
# for i, next_df in enumerate(tqdm(dataframes[1:], desc="Unioning DataFrames", unit="df", initial=2, total=len(dataframes)), start=2):
#     df = df.unionByName(next_df)

print("[2/6] Extracting and indexing paragraphs")
paragraphs_df = df.select(col("id").cast("long"), "content")

paragraphs_df.persist()

print("[3/6] Creating (anchor, positive) pairs")
anchor_df = paragraphs_df.alias("a")
positive_df = paragraphs_df.alias("p")

adjacent_pairs = anchor_df.join(
    positive_df,
    col("a.id") + 1 == col("p.id")
).select(
    col("a.content").alias("anchor"),
    col("p.content").alias("positive")
)

adjacent_pairs = adjacent_pairs \
    .withColumn("pair_index", monotonically_increasing_id())

print("[4/6] Preparing shuffled negatives")
negatives_df = paragraphs_df.select(col("content").alias("negative")) \
    .withColumn("neg_index", monotonically_increasing_id())

print("[5/6] Joining anchor-positive with random negatives")
triplets = adjacent_pairs.join(
    negatives_df,
    adjacent_pairs["pair_index"] == negatives_df["neg_index"]
).select("anchor", "positive", "negative")

print(f"[6/6] Writing triplets to Spark part files: {OUTPUT_DIR}")
triplets.write.mode("overwrite").json(OUTPUT_DIR)

print(f"Training triplets saved to Spark part files: {OUTPUT_DIR}")

# Stop Spark
spark.stop()
