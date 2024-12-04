from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.sql.functions import trim, from_csv, col, avg, year, weekofyear, to_timestamp, count

# Define the schema based on CSV
csv_schema = StructType([
    StructField("STATION", StringType(), True),
    StructField("DATE", StringType(), True),
    StructField("LATITUDE", StringType(), True),  # Initially StringType for trimming
    StructField("LONGITUDE", StringType(), True), # Initially StringType for trimming
    StructField("ELEVATION", StringType(), True), # Initially StringType for trimming
    StructField("NAME", StringType(), True),
    StructField("TEMP", StringType(), True),      # Initially StringType for trimming
    StructField("TEMP_ATTRIBUTES", StringType(), True),
    StructField("DEWP", StringType(), True),      # Initially StringType for trimming
    StructField("DEWP_ATTRIBUTES", StringType(), True),
    StructField("SLP", StringType(), True),       # Initially StringType for trimming
    StructField("SLP_ATTRIBUTES", StringType(), True),
    StructField("STP", StringType(), True),       # Initially StringType for trimming
    StructField("STP_ATTRIBUTES", StringType(), True),
    StructField("VISIB", StringType(), True),     # Initially StringType for trimming
    StructField("VISIB_ATTRIBUTES", StringType(), True),
    StructField("WDSP", StringType(), True),      # Initially StringType for trimming
    StructField("WDSP_ATTRIBUTES", StringType(), True),
    StructField("MXSPD", StringType(), True),     # Initially StringType for trimming
    StructField("GUST", StringType(), True),      # Initially StringType for trimming
    StructField("MAX", StringType(), True),       # Initially StringType for trimming
    StructField("MAX_ATTRIBUTES", StringType(), True),
    StructField("MIN", StringType(), True),       # Initially StringType for trimming
    StructField("MIN_ATTRIBUTES", StringType(), True),
    StructField("PRCP", StringType(), True),      # Initially StringType for trimming
    StructField("PRCP_ATTRIBUTES", StringType(), True),
    StructField("SNDP", StringType(), True),      # Initially StringType for trimming
    StructField("FRSHTT", StringType(), True)
])

def trim_and_cast(df):
    # Trim all string columns
    for field in csv_schema.fields:
        if isinstance(field.dataType, StringType):
            df = df.withColumn(field.name, trim(col(field.name)))

    # Cast FloatType columns back to their types
    float_columns = ["LATITUDE", "LONGITUDE", "ELEVATION", "TEMP", "DEWP", 
                     "SLP", "STP", "VISIB", "WDSP", "MXSPD", "GUST", 
                     "MAX", "MIN", "PRCP", "SNDP"]
    for col_name in float_columns:
        df = df.withColumn(col_name, col(col_name).cast(FloatType()))
    
    return df

def remove_invalid_rows(df):
    """
    Remove rows where `DATE` or `STATION` is null, ensuring valid data is processed.
    """
    return df.filter(col("DATE").isNotNull() & col("STATION").isNotNull())

def read_and_process_from_kafka(kafka_topic, hdfs_output_dir, bootstrap_servers='localhost:9092'):
    # Create a Spark session
    spark = SparkSession.builder \
        .appName("KafkaToHDFS") \
        .getOrCreate()
    
    # Read data from Kafka
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", bootstrap_servers) \
        .option("subscribe", kafka_topic) \
        .option("startingOffsets", "latest") \
        .load()

    # Convert Kafka value column to string and parse as CSV using the schema
    csv_df = df.selectExpr("CAST(value AS STRING) AS csv_value") \
                .select(from_csv(col("csv_value"), csv_schema.simpleString()).alias("parsed")) \
                .select("parsed.*")
    
    # Trim and cast the columns
    csv_df_cleaned_1 = trim_and_cast(csv_df).filter(col("STATION") != "STATION")
    csv_df_cleaned = remove_invalid_rows(csv_df_cleaned_1)  # Filter out invalid rows

    # Ensure DATE is parsed as a timestamp
    csv_df_with_timestamp = csv_df.withColumn("DATE", to_timestamp(col("DATE"), "yyyy-MM-dd"))

    # Apply watermark on the timestamp column
    df_with_watermark = csv_df_with_timestamp.withWatermark("DATE", "7 days")

    # Step 2: Add Year and Week columns
    df_with_week = df_with_watermark.withColumn("Weather_Year", year(col("DATE"))) \
                                    .withColumn("Weather_Week", weekofyear(col("DATE")))

    # Step 3: Group and calculate averages
    df_weekly_avg = df_with_week.groupBy("Weather_Year", "Weather_Week", "DATE").agg(
        avg("LATITUDE").alias("avg_LATITUDE"),
        avg("LONGITUDE").alias("avg_LONGITUDE"),
        avg("ELEVATION").alias("avg_ELEVATION"),
        avg("TEMP").alias("avg_TEMP"),
        avg("DEWP").alias("avg_DEWP"),
        avg("SLP").alias("avg_SLP"),
        avg("STP").alias("avg_STP"),
        avg("VISIB").alias("avg_VISIB"),
        avg("WDSP").alias("avg_WDSP"),
        avg("MXSPD").alias("avg_MXSPD"),
        avg("GUST").alias("avg_GUST"),
        avg("MAX").alias("avg_MAX"),
        avg("MIN").alias("avg_MIN"),
        avg("PRCP").alias("avg_PRCP"),
        avg("SNDP").alias("avg_SNDP"),
        avg("FRSHTT").alias("avg_FRSHTT"),
        count("*").alias("record_count")
    )

    # Step 4: Write to HDFS in Parquet format
    hdfs_query = df_weekly_avg.writeStream \
        .outputMode("append") \
        .format("parquet") \
        .option("path", hdfs_output_dir) \
        .partitionBy("Weather_Year") \
        .option("checkpointLocation", f"{hdfs_output_dir}/_checkpoint") \
        .start()

    hdfs_query.awaitTermination()

# Call the function
read_and_process_from_kafka("climate-sample", "/climate/processed_data")