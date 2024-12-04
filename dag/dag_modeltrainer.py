from airflow import DAG
from airflow.operators.python import PythonOperator
from pyspark.sql.window import Window
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import logging
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, log1p, expm1, abs, sin, cos, lit
from pyspark.sql.window import Window
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import log1p, expm1, abs, col
from pyspark.sql.window import Window
from pyspark.ml.regression import RandomForestRegressor
from functools import reduce
from influxdb import InfluxDBClient
from datetime import datetime
import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from pyspark.sql import functions as F
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_FILES_LOG = "/tmp/processed_parquet_files.log"
BASE_PATH = "/climate/processed_data"
BASE_WEATHER_PATH = "/climate/yearwise_data"


# Define the schema explicitly (adjust fields as needed for your use case)
empty_schema = StructType([
    StructField("Weather_Week", StringType(), True),
    StructField("record_count", IntegerType(), True),
    StructField("avg_LATITUDE", DoubleType(), True),
    StructField("avg_LONGITUDE", DoubleType(), True),
    StructField("avg_ELEVATION", DoubleType(), True),
    StructField("avg_TEMP", DoubleType(), True),
    StructField("avg_DEWP", DoubleType(), True),
    StructField("avg_SLP", DoubleType(), True),
    StructField("avg_STP", DoubleType(), True),
    StructField("avg_VISIB", DoubleType(), True),
    StructField("avg_WDSP", DoubleType(), True),
    StructField("avg_MXSPD", DoubleType(), True),
    StructField("avg_GUST", DoubleType(), True),
    StructField("avg_MAX", DoubleType(), True),
    StructField("avg_MIN", DoubleType(), True),
    StructField("avg_PRCP", DoubleType(), True),
    StructField("avg_SNDP", DoubleType(), True),
    StructField("avg_FRSHTT", DoubleType(), True),
])

# Default arguments for Airflow DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Airflow DAG
with DAG(
    "dsproject_train_weather_data",
    default_args=default_args,
    description="A DAG to process weather data and pass differing records via XCom",
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    def push_predictions_to_influxdb(predictions_df):
        """
        Push prediction details to InfluxDB under the `predicted_cases` measurement.
        """
        # Get the InfluxDB token
        token = "104BlY52IbebcrweMnhQ4p0QgUc0usKiqbSTYnfBEXP9dLasdflzPYpsctbFd7JUwddYdHhAhKHeRKy7t4goAQ=="  # Replace with your InfluxDB token
        if not token:
            raise ValueError("InfluxDB token is not set.")

        org = "Project"  # Replace with your organization name
        bucket = "Prediction"  # Replace with your bucket name
        url = "http://localhost:8086"  # Replace with your InfluxDB URL

        # Initialize InfluxDB client
        write_client = InfluxDBClient(url=url, token=token, org=org)
        write_api = write_client.write_api(write_options=SYNCHRONOUS)

        # Compute the first date of the week
        df_with_first_date = predictions_df.withColumn(
            "week_start_date",
            F.date_add(
                F.to_date(F.concat(F.col("year"), F.lit("-01-01")), "yyyy-MM-dd"),
                (F.col("weekofyear") - 1) * 7  # Adjust by weeks
            )
        ).withColumn(
            "time", F.col("week_start_date").cast("timestamp")  # Ensure `time` is a timestamp
        )

        # Collect rows with the `time` field
        for row in df_with_first_date.select("year", "weekofyear", "Prediction Cases", "time").collect():
            point = (
                Point("predicted_cases")
                .tag("year", str(row["year"]))  # Tag year as a string
                .field("weekofyear", int(row["weekofyear"]))  # Week number as a field
                .field("predicted_cases", float(row["Prediction Cases"]))  # Predicted cases as a field
                .time(row["time"].isoformat(), WritePrecision.NS)  # Use `time` from the row
            )
            write_api.write(bucket=bucket, org=org, record=point)

        write_client.close()
        print("Predictions successfully pushed to InfluxDB.")

    def push_accuracy_to_influxdb(accuracy, mape):
        """
        Push accuracy metrics to InfluxDB under the `model_accuracy` measurement.
        """
        # Get the InfluxDB token from environment variables
        token = "104BlY52IbebcrweMnhQ4p0QgUc0usKiqbSTYnfBEXP9dLasdflzPYpsctbFd7JUwddYdHhAhKHeRKy7t4goAQ=="
        if not token:
            raise ValueError("InfluxDB token is not set in the environment variables.")

        org = "Project"  # Replace with your organization name
        bucket = "Accuracy"  # Replace with your bucket name
        url = "http://localhost:8086"  # Replace with your InfluxDB URL

        # Initialize InfluxDB client
        write_client = InfluxDBClient(url=url, token=token, org=org)
        write_api = write_client.write_api(write_options=SYNCHRONOUS)

        # Create a point for accuracy metrics
        point = (
            Point("model_accuracy")
            .field("accuracy", float(accuracy))  # Accuracy as a field
            .field("mape", float(mape))          # MAPE as a field
            .time(datetime.utcnow())            # Current UTC time
        )

        # Write the point to InfluxDB
        write_api.write(bucket=bucket, org=org, record=point)
        write_client.close()
        print("Accuracy metrics successfully pushed to InfluxDB.")

    def update_processed_files_log(log_path, new_files):
        """
        Append new processed files to the log file.
        """
        logger.info("Updating processed files log.")
        try:
            with open(log_path, "a") as f:
                for file in new_files:
                    f.write(file + "\n")
            logger.info("Processed files log updated successfully.")
        except Exception as e:
            logger.error(f"Error updating log file: {e}")
            raise


    def get_year_from_path(file_path):
        """
        Extract the year from the file path, assuming folder structure includes 'Weather_Year=YYYY'.
        """
        parts = file_path.split("/")
        for part in parts:
            if part.startswith("Weather_Year="):
                return part.split("=")[-1]
        return None
    
    def path_exists_in_hdfs(path, spark):
        """
        Check if a given path exists in HDFS.
        """
        try:
            fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
            hdfs_path = spark._jvm.org.apache.hadoop.fs.Path(path)
            return fs.exists(hdfs_path)
        except Exception as e:
            logger.error(f"Error checking path in HDFS: {e}")
            return False

    def list_files_to_process(base_path, **kwargs):
        """
        List all Parquet files under the base HDFS path, segregated by year.
        """
        ti = kwargs.get("ti", None)
        if not ti:
            logger.error("TaskInstance (ti) is None. XCom operations cannot proceed.")
            return []

        spark = SparkSession.builder.appName("ListFiles").getOrCreate()
        files_to_process = []
        try:
            fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
            base_path_obj = spark._jvm.org.apache.hadoop.fs.Path(base_path)

            if fs.exists(base_path_obj):
                file_status_list = fs.listStatus(base_path_obj)
                for file_status in file_status_list:
                    if file_status.isDirectory():
                        # Recursive call and merging results
                        sub_files = list_files_to_process(file_status.getPath().toString(), **kwargs)
                        if sub_files:
                            files_to_process.extend(sub_files)
                    elif file_status.getPath().toString().endswith(".parquet"):
                        files_to_process.append(file_status.getPath().toString())

                logger.info(f"Found files: {files_to_process}")
            else:
                logger.warning(f"Path does not exist: {base_path}")

            # Push the list of files to XCom
            ti.xcom_push(key="files_to_process", value=files_to_process)
            logger.info(f"Files pushed to XCom: {files_to_process}")
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            raise
        return files_to_process  # Ensure the function always returns a list

    def process_files(**kwargs):
        """
        Process weather data files and find differing records.
        """
        ti = kwargs.get("ti", None)
        if not ti:
            logger.error("TaskInstance (ti) is None. Cannot pull files from XCom.")
            return

        spark = SparkSession.builder.appName("ProcessWeatherData").getOrCreate()
        files_to_process = ti.xcom_pull(task_ids="list_files_to_process", key="files_to_process")
        if not files_to_process:
            logger.warning("No files to process. Skipping processing step.")
            return

        processed_files = []
        updated_years = []

        for file_path in files_to_process:
            year = get_year_from_path(file_path)
            if not year:
                logger.warning(f"Skipping file {file_path} - Year not found in path.")
                continue

            weather_data_path = f"{BASE_WEATHER_PATH}/Weather_Year={year}"
            try:
                logger.info(f"Processing file: {file_path}")
                new_data = spark.read.parquet(file_path)

                if path_exists_in_hdfs(weather_data_path, spark):
                    existing_data = spark.read.parquet(weather_data_path)
                    merged_data = (
                        existing_data.unionByName(new_data, allowMissingColumns=True)
                        .groupBy("Weather_Week")
                        .agg(
                            F.sum("record_count").alias("record_count"),
                            *[
                                (F.sum(F.col(c) * F.col("record_count")) / F.sum("record_count")).alias(c)
                                for c in [
                                    "avg_LATITUDE", "avg_LONGITUDE", "avg_ELEVATION",
                                    "avg_TEMP", "avg_DEWP", "avg_SLP", "avg_STP",
                                    "avg_VISIB", "avg_WDSP", "avg_MXSPD", "avg_GUST",
                                    "avg_MAX", "avg_MIN", "avg_PRCP", "avg_SNDP", "avg_FRSHTT"
                                ]
                            ]
                        )
                    )
                else:
                    logger.info(f"No existing data found for year {year}. Writing new data.")
                    existing_data = spark.createDataFrame([], schema=empty_schema)
                    merged_data = (
                        existing_data.unionByName(new_data, allowMissingColumns=True)
                        .groupBy("Weather_Week")
                        .agg(
                            F.sum("record_count").alias("record_count"),
                            *[
                                (F.sum(F.col(c) * F.col("record_count")) / F.sum("record_count")).alias(c)
                                for c in [
                                    "avg_LATITUDE", "avg_LONGITUDE", "avg_ELEVATION",
                                    "avg_TEMP", "avg_DEWP", "avg_SLP", "avg_STP",
                                    "avg_VISIB", "avg_WDSP", "avg_MXSPD", "avg_GUST",
                                    "avg_MAX", "avg_MIN", "avg_PRCP", "avg_SNDP", "avg_FRSHTT"
                                ]
                            ]
                        )
                    )
                merged_data.write.mode("overwrite").parquet(weather_data_path)
                processed_files.append(file_path)
                updated_years.append(year)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue                    
        updated_years = sorted(set(updated_years))
        update_processed_files_log(PROCESSED_FILES_LOG, processed_files)
        ti.xcom_push(key="differing_records", value=updated_years)
        logger.info("Differing records pushed to XCom.")

    # Replace the `load_differing_records` function with the new logic

    def load_differing_records_and_train(**kwargs):
        """
        Process weather data, separate matching and non-matching records with dengue data,
        train using matching records, update the model, and store predictions for non-matching records.
        """
        ti = kwargs.get("ti", None)
        if not ti:
            logger.error("TaskInstance (ti) is None. Cannot pull differing records from XCom.")
            return
        
        spark = SparkSession.builder.appName("LoadAndTrainDengueData").getOrCreate()

        # Paths for data and model storage in HDFS
        train_data_path = "hdfs://localhost:9000/climate/model/train_dataset"
        test_data_path = "hdfs://localhost:9000/climate/model/test_dataset"
        model_path = "hdfs://localhost:9000/climate/model/final_dengue_model"
        prediction_output_path = "hdfs://localhost:9000/climate/model/prediction_output"

        # Load the pre-trained model from HDFS
        from pyspark.ml.regression import RandomForestRegressionModel
        try:
            model = RandomForestRegressionModel.load(model_path)
            logger.info("Pre-trained model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load the model from {model_path}: {e}")
            return

        # Load existing train and test datasets
        try:
            train_df = spark.read.parquet(train_data_path)
            test_df = spark.read.parquet(test_data_path)
            logger.info("Train and test datasets loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load train/test datasets: {e}")
            return

        updated_years = ti.xcom_pull(task_ids="process_files", key="differing_records")
        if not updated_years:
            logger.warning("No updated years found. Skipping processing step.")
            return
            
        # Load dengue data
        dengue_data_path = "/climate/raw/denguedata/DengAI_Predicting_Disease_Spread_-_Training_Data_Labels.csv"
        df_dengue = spark.read.csv(dengue_data_path, header=True, inferSchema=True).dropna()
        df_dengue = df_dengue.filter(df_dengue["city"] == "sj")

        window_spec = Window.partitionBy("city").orderBy("year", "weekofyear")
        df_dengue = df_dengue.withColumn("lag_cases", F.lag("total_cases", 1).over(window_spec))
        df_dengue = df_dengue.withColumn("lag_cases_2", F.lag("total_cases", 2).over(window_spec))
        df_dengue = df_dengue.withColumn("lag_cases_3", F.lag("total_cases", 3).over(window_spec))
        df_dengue = df_dengue.withColumn("sin_week", F.sin(2 * F.lit(3.141592653589793) * df_dengue["weekofyear"] / 52)) \
                    .withColumn("cos_week", F.cos(2 * F.lit(3.141592653589793) * df_dengue["weekofyear"] / 52))

        for year in updated_years:
            weather_data_path = f"/climate/yearwise_data/Weather_Year={year}"
            if path_exists_in_hdfs(weather_data_path, spark):
                logger.info(f"Loading weather data for year {year}")
                df_weather = spark.read.parquet(weather_data_path)
                df_weather = df_weather.withColumn("Weather_Year", F.lit(year))

                df_dengue_filtered = df_dengue.filter(F.col("year") == year)
                # Join dengue and weather data
                df_joined = df_dengue_filtered.join(
                    df_weather,
                    (df_dengue["year"] == df_weather["Weather_Year"]) &
                    (df_dengue["weekofyear"] == df_weather["Weather_Week"]),
                    "left"
                )

                # Step 6: Add Interaction Features
                df_joined = df_joined.withColumn("temp_prcp_interaction", col("avg_TEMP") * col("avg_PRCP"))

                # Step 7: Handle Null Values and Cast Columns
                columns_to_cast = ["year", "weekofyear", "avg_TEMP", "avg_PRCP", "avg_DEWP", "lag_cases", "lag_cases_2", "lag_cases_3", "sin_week", "cos_week", "temp_prcp_interaction"]
                for col_name in columns_to_cast:
                    df_joined = df_joined.withColumn(col_name, F.col(col_name).cast(DoubleType()))
                df_joined = df_joined.fillna({col_name: 0 for col_name in columns_to_cast})


                # Separate matching and non-matching records
                matching_records = df_joined.filter(df_joined["Weather_Year"].isNotNull())
                non_matching_records = df_joined.filter(df_joined["Weather_Year"].isNull())
                matching_records = matching_records.fillna(0).withColumn("log_total_cases", log1p(col("total_cases")))
                # Extract distinct year and weekofyear combinations
                if matching_records.count() != 0:
                    max_year_week = matching_records.select(
                        F.max("year").alias("max_year"),
                        F.max("weekofyear").alias("max_week")
                    ).collect()[0]

                    max_year = max_year_week["max_year"]
                    max_week = max_year_week["max_week"]

                    logger.info(f"Max year: {max_year}, Max week: {max_week}")

                    # Step 2: Filter non_matching_records for combinations exceeding the max year-week
                    non_matching_exceeding = non_matching_records.filter(
                        (F.col("year") > max_year) |
                        ((F.col("year") == max_year) & (F.col("weekofyear") > max_week))
                    )

                    # Step 3: Collect the filtered combinations as an array
                    non_matching_exceeding_array = non_matching_exceeding.select("year", "weekofyear") \
                        .distinct() \
                        .rdd.map(lambda row: (int(row["year"]), int(row["weekofyear"]))) \
                        .collect()

                    ti.xcom_push(key="non_matching_combinations", value=non_matching_exceeding_array)
                    logger.info(f"Non-matching combinations passed via XCom: {non_matching_exceeding_array}")

                # Train the model using matching records
                if matching_records.count() > 0:
                    # Step 1: Get column sets from both DataFrames
                    training_columns = set(train_df.columns)
                    matching_columns = set(matching_records.columns)

                    # Step 2: Drop extra columns from the matching records
                    extra_columns = matching_columns - training_columns
                    if extra_columns:
                        logger.info(f"Dropping extra columns from matching records: {extra_columns}")
                        matching_records = matching_records.drop(*extra_columns)

                    # Step 3: Add missing columns to the matching records
                    missing_columns = training_columns - matching_columns
                    if missing_columns:
                        logger.info(f"Adding missing columns to matching records: {missing_columns}")
                        for col_name in missing_columns:
                            # Add with default values (e.g., 0 for numeric columns)
                            matching_columns = matching_columns.withColumn(col_name, F.lit(None).cast(train_df.schema[col_name].dataType))

                    # Step 4: Ensure column order matches (optional but recommended)
                    matching_records = matching_records.select(*train_df.columns)

                    matching_records.printSchema()
                    train_df = train_df.unionByName(matching_records)
                    writable_train_df = train_df
                    writable_train_df = writable_train_df.fillna(0).withColumn("log_total_cases", log1p(col("total_cases")))

                    feature_columns = [
                        "year", "weekofyear", "avg_TEMP", "avg_PRCP", "avg_DEWP",
                        "lag_cases", "lag_cases_2", "lag_cases_3", "sin_week", "cos_week",
                        "temp_prcp_interaction"
                    ]
                    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
                    feature_train_df = assembler.transform(writable_train_df)
                    feature_train_df.printSchema()

                    logger.info("Matching records added to the train dataset.")

                    # Step 3: Define the regressor and retrain the model
                    rf = RandomForestRegressor(
                        featuresCol="features",
                        labelCol="log_total_cases",
                        predictionCol="log_prediction",
                        numTrees=100,
                        maxDepth=10
                    )
                    updated_model = rf.fit(feature_train_df)
                    logger.info("Model retrained with combined dataset.")

                    # Save the updated model back to HDFS
                    updated_model.write().overwrite().save(model_path)
                    logger.info(f"Updated model saved to {model_path}.")

                    # Save the updated train dataset back to HDFS
                    train_df.write.mode("append").parquet(train_data_path)
                    logger.info(f"Updated train dataset saved to {train_data_path}.")
                else:
                    logger.warning(f"No matching records found for year {year}. Skipping training.")

                # Predict for non-matching records
                if non_matching_records.count() > 0:
                    df_weather = df_weather.withColumn("year", F.col("Weather_Year").cast("int")) \
                        .withColumn("weekofyear", F.col("Weather_Week").cast("int"))
                    logger.info("No matching records found. Considering all year-week combinations from weather data.")
                    # Extract the distinct year and weekofyear combinations from df_weather
                    non_matching_exceeding_array = [
                        (int(row["year"]), int(row["weekofyear"]))
                        for row in df_weather.select("year", "weekofyear").distinct().collect()
                    ]
                    ti.xcom_push(key="non_matching_combinations", value=non_matching_exceeding_array)
                    logger.info(f"Predictions for {non_matching_exceeding_array} will be processed.")
                else:
                    logger.warning(f"No non-matching records found for year {year}. Skipping predictions.")
            else:
                logger.warning(f"Weather data for year {year} not found at {weather_data_path}.")

    # Replace the `load_differing_records` function with the new logic
    def test_model_accuracy():
        # Define paths
        test_data_path = "hdfs://localhost:9000/climate/model/test_dataset"
        model_path = "hdfs://localhost:9000/climate/model/final_dengue_model"
        spark = SparkSession.builder.appName("TestDengueData").getOrCreate()
        # Step 1: Fetch the pre-trained model from HDFS
        try:
            model = RandomForestRegressionModel.load(model_path)
            logger.info("Model loaded successfully from HDFS.")
        except Exception as e:
            logger.error(f"Failed to load model from HDFS: {e}")
            raise

        # Step 2: Load the training dataset from HDFS
        try:
            test_df = spark.read.parquet(test_data_path)
            logger.info("Testing dataset loaded successfully from HDFS.")
        except Exception as e:
            logger.error(f"Failed to load training dataset from HDFS: {e}")
            raise

        # Step 3: Prepare the training dataset for evaluation
        # Define feature columns
        feature_columns = [
            "year", "weekofyear", "avg_TEMP", "avg_PRCP", "avg_DEWP",
            "lag_cases", "lag_cases_2", "lag_cases_3", "sin_week", "cos_week", "temp_prcp_interaction"
        ]

        # Add the log-transformed target column
        test_df = test_df.withColumn("log_total_cases", log1p(F.col("total_cases")))

        # Assemble features into a vector
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        test_df = assembler.transform(test_df)

        # Step 4: Make predictions using the loaded model
        predictions = model.transform(test_df)

        # Reverse the log transformation for predictions
        predictions = predictions.withColumn("Prediction Cases", expm1(F.col("log_prediction")))

        # Step 5: Evaluate the Model
        # R2 Score
        evaluator = RegressionEvaluator(labelCol="total_cases", predictionCol="Prediction Cases", metricName="r2")
        r2_score = evaluator.evaluate(predictions)
        logger.info(f"R2 Score: {r2_score}")
        print(f"R2 Score: {r2_score}")

        # MAPE (Mean Absolute Percentage Error)
        mape_df = predictions.withColumn("ape", abs((col("total_cases") - col("Prediction Cases")) / col("total_cases")))
        mape = mape_df.selectExpr("avg(ape) as mape").first()["mape"] * 100
        accuracy = 100 - mape
        logger.info(f"MAPE: {mape:.2f}%")
        logger.info(f"Accuracy: {accuracy:.2f}%")
        push_accuracy_to_influxdb(accuracy, mape)
        print(f"MAPE: {mape:.2f}%")
        print(f"Accuracy: {accuracy:.2f}%")

    def predict_cases_from_combinations(**kwargs):
        """
        Predict cases using the model and year-week combinations from XCom.
        """

        ti = kwargs.get("ti", None)
        if not ti:
            logger.error("TaskInstance (ti) is None. Cannot pull differing records from XCom.")
            return
        year_week_combinations = ti.xcom_pull(key="non_matching_combinations", task_ids="load_differing_records_and_train")
    
        if not year_week_combinations:
            logger.error("No year-week combinations retrieved from XCom.")
            return
        
        spark = SparkSession.builder.appName("PredictCases").getOrCreate()
        
        if not year_week_combinations:
            logger.error("No year-week combinations retrieved from XCom.")
            return
        
        # Load model from HDFS
        model_path = "hdfs://localhost:9000/climate/model/final_dengue_model"
        try:
            model = RandomForestRegressionModel.load(model_path)
            logger.info("Model loaded successfully from HDFS.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return

        # Load weather data from HDFS
        weather_data_path = "hdfs://localhost:9000/climate/yearwise_data"
        try:
            df_weather = spark.read.parquet(weather_data_path)
            df_weather = df_weather.withColumn("Weather_Year", col("Weather_Year").cast("int")) \
                                .withColumn("Weather_Week", col("Weather_Week").cast("int"))
            logger.info("Weather data loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load weather data: {e}")
            return

        # Filter weather data for the given year-week combinations
        year_week_conditions = [
            (col("Weather_Year") == year) & (col("Weather_Week") == week)
            for year, week in year_week_combinations
        ]
        filtered_weather = df_weather.filter(reduce(lambda x, y: x | y, year_week_conditions))

        # Add derived features
        filtered_weather = filtered_weather.withColumn("sin_week", sin(2 * lit(3.141592653589793) * col("Weather_Week") / 52)) \
                                        .withColumn("cos_week", cos(2 * lit(3.141592653589793) * col("Weather_Week") / 52)) \
                                        .withColumn("temp_prcp_interaction", col("avg_TEMP") * col("avg_PRCP")) \
                                        .withColumnRenamed("Weather_Year", "year") \
                                        .withColumnRenamed("Weather_Week", "weekofyear") \
                                        .withColumn("lag_cases", lit(0.0)) \
                                        .withColumn("lag_cases_2", lit(0.0)) \
                                        .withColumn("lag_cases_3", lit(0.0))

        # Assemble features
        feature_columns = ["year", "weekofyear", "avg_TEMP", "avg_PRCP", "avg_DEWP", 
                        "lag_cases", "lag_cases_2", "lag_cases_3", "sin_week", 
                        "cos_week", "temp_prcp_interaction"]
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        testDf = assembler.transform(filtered_weather)

        # Predict cases
        predictions = model.transform(testDf)
        predictions = predictions.withColumn("Prediction Cases", expm1(col("log_prediction")))

        # Save predictions
        predictions_path = "hdfs://localhost:9000/climate/predictions"
        # Print predictions with only relevant columns
        predictions.select("year", "weekofyear", "Prediction Cases").show(truncate=False)
        predictions.select("year", "weekofyear", "Prediction Cases").write.mode("overwrite").parquet(predictions_path)
        push_predictions_to_influxdb(predictions)
        logger.info(f"Predictions saved to {predictions_path}.")


    # Tasks
    list_files_task = PythonOperator(
        task_id="list_files_to_process",
        python_callable=list_files_to_process,
        provide_context=True,
        op_kwargs={"base_path": "/climate/processed_data"},
        dag=dag,
    )

    process_files_task = PythonOperator(
        task_id="process_files",
        python_callable=process_files,
        provide_context=True,
        dag=dag
    )

    load_differing_records_and_train_task = PythonOperator(
        task_id="load_differing_records_and_train",
        python_callable=load_differing_records_and_train,
        provide_context=True,
        dag=dag
    )

    test_model_accuracy_task = PythonOperator(
        task_id="test_model_accuracy",
        python_callable=test_model_accuracy,
        provide_context=True,
        dag=dag
    )

    predict_cases_from_combinations_task = PythonOperator(
        task_id="predict_cases_from_combinations",
        python_callable=predict_cases_from_combinations,
        provide_context=True,
        dag=dag
    )
    list_files_task >> process_files_task >> load_differing_records_and_train_task >> test_model_accuracy_task >> predict_cases_from_combinations_task
