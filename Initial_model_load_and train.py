from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, abs, sin, cos, log1p, expm1
from pyspark.sql.window import Window
from pyspark.sql.types import ArrayType
from influxdb import InfluxDBClient
from datetime import datetime
import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from pyspark.sql import functions as F
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime

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

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("DenguePrediction") \
    .config("spark.executor.memory", "2g") \
    .getOrCreate()

# Define HDFS paths
climate_data_path = "hdfs://localhost:9000/climate/raw/72494523293.csv-*"
dengue_data_path = "hdfs://localhost:9000/climate/raw/denguedata/DengAI_Predicting_Disease_Spread_-_Training_Data_Labels.csv"
train_output_path = "hdfs://localhost:9000/climate/model/train_dataset"
test_output_path = "hdfs://localhost:9000/climate/model/test_dataset"
model_output_path = "hdfs://localhost:9000/climate/model/final_dengue_model"

# Step 1: Load Climate Data
df_weather = spark.read.csv(climate_data_path, header=True, inferSchema=True).dropna()
df_weather = df_weather.withColumn("Weather_Year", F.year(df_weather["DATE"])) \
                       .withColumn("Weather_Week", F.weekofyear(df_weather["DATE"]))

# Step 2: Calculate Weekly Averages
df_weekly_avg = df_weather.groupBy("Weather_Year", "Weather_Week").agg(
    F.avg("LATITUDE").alias("avg_LATITUDE"),
    F.avg("LONGITUDE").alias("avg_LONGITUDE"),
    F.avg("ELEVATION").alias("avg_ELEVATION"),
    F.avg("TEMP").alias("avg_TEMP"),
    F.avg("DEWP").alias("avg_DEWP"),
    F.avg("SLP").alias("avg_SLP"),
    F.avg("STP").alias("avg_STP"),
    F.avg("VISIB").alias("avg_VISIB"),
    F.avg("WDSP").alias("avg_WDSP"),
    F.avg("MXSPD").alias("avg_MXSPD"),
    F.avg("GUST").alias("avg_GUST"),
    F.avg("MAX").alias("avg_MAX"),
    F.avg("MIN").alias("avg_MIN"),
    F.avg("PRCP").alias("avg_PRCP"),
    F.avg("SNDP").alias("avg_SNDP"),
    F.avg("FRSHTT").alias("avg_FRSHTT")
)

# Step 3: Load Dengue Data
df_den = spark.read.csv(dengue_data_path, header=True, inferSchema=True).dropna()
df_den = df_den.filter(df_den['city'] == 'sj')

# Step 4: Add Lag Features and Seasonal Features
window_spec = Window.partitionBy("city").orderBy("year", "weekofyear")
df_den = df_den.withColumn("lag_cases", F.lag("total_cases", 1).over(window_spec))
df_den = df_den.withColumn("lag_cases_2", F.lag("total_cases", 2).over(window_spec))
df_den = df_den.withColumn("lag_cases_3", F.lag("total_cases", 3).over(window_spec))
df_den = df_den.withColumn("sin_week", F.sin(2 * F.lit(3.141592653589793) * df_den["weekofyear"] / 52)) \
               .withColumn("cos_week", F.cos(2 * F.lit(3.141592653589793) * df_den["weekofyear"] / 52))

# Step 5: Join Dengue and Climate Data
df_joined = df_den.join(
    df_weekly_avg,
    (df_den.year == df_weekly_avg.Weather_Year) & (df_den.weekofyear == df_weekly_avg.Weather_Week),
    "inner"
).drop(df_weekly_avg["Weather_Year"]).drop(df_weekly_avg["Weather_Week"])

# Step 6: Add Interaction Features
df_joined = df_joined.withColumn("temp_prcp_interaction", col("avg_TEMP") * col("avg_PRCP"))

# Step 7: Handle Null Values and Cast Columns
columns_to_cast = ["year", "weekofyear", "avg_TEMP", "avg_PRCP", "avg_DEWP", "lag_cases", "lag_cases_2", "lag_cases_3", "sin_week", "cos_week", "temp_prcp_interaction"]
for col_name in columns_to_cast:
    df_joined = df_joined.withColumn(col_name, F.col(col_name).cast(DoubleType()))
df_joined = df_joined.fillna({col_name: 0 for col_name in columns_to_cast})

# Step 8: Train-Test Split
train, test = df_joined.randomSplit([0.8, 0.2], seed=1947)

# Save Train and Test Datasets
train.write.mode("overwrite").parquet(train_output_path)
test.write.mode("overwrite").parquet(test_output_path)

# Step 9: Feature Engineering and Log Transformation
feature_columns = ["year", "weekofyear", "avg_TEMP", "avg_PRCP", "avg_DEWP", "lag_cases", "lag_cases_2", "lag_cases_3", "sin_week", "cos_week", "temp_prcp_interaction"]
train = train.withColumn("log_total_cases", log1p(F.col("total_cases")))
test = test.withColumn("log_total_cases", log1p(F.col("total_cases")))

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
trainDf = assembler.transform(train)
trainDf.show(5)

# Step 10: Train a Random Forest Regressor
rf = RandomForestRegressor(featuresCol="features", labelCol="log_total_cases", predictionCol="log_prediction", numTrees=100, maxDepth=10, maxBins=32)
model = rf.fit(trainDf)

# Save the Trained Model
model.write().overwrite().save(model_output_path)

# Step 11: Prepare Test Data and Make Predictions
testDf = assembler.transform(test)
testDf.show(5)


predictions = model.transform(testDf)

# Reverse the Log Transformation for Predictions
predictions = predictions.withColumn("Prediction Cases", expm1(F.col("log_prediction")))

# Step 12: Evaluate the Model
evaluator = RegressionEvaluator(labelCol="total_cases", predictionCol="Prediction Cases", metricName="r2")
r2 = evaluator.evaluate(predictions)
print(f"R2: {r2}")

# Calculate MAPE (Mean Absolute Percentage Error)
mape_df = predictions.withColumn("ape", abs((col("total_cases") - col("Prediction Cases")) / col("total_cases")))
mape = mape_df.selectExpr("avg(ape) as mape").first()["mape"] * 100
print(f"MAPE: {mape:.2f}%")
print(f"Accuracy: {100 - mape:.2f}%")
push_accuracy_to_influxdb(mape,100-mape)
# Step 13: Predict for Future Climate Data
print("Loading future climate data...")
future_climate_data_path = "hdfs://localhost:9000/climate/raw/72494523293.csv-2024"
future_weather = spark.read.csv(future_climate_data_path, header=True, inferSchema=True).dropna()
print(f"Future climate data loaded with {future_weather.count()} rows")
future_weather.printSchema()

# Extract Year and Week
future_weather = future_weather.withColumn("Weather_Year", F.year(future_weather["DATE"])) \
                               .withColumn("Weather_Week", F.weekofyear(future_weather["DATE"]))

# Calculate Weekly Averages
print("Calculating weekly averages for future data...")
future_weekly_avg = future_weather.groupBy("Weather_Year", "Weather_Week").agg(
    F.avg("LATITUDE").alias("avg_LATITUDE"),
    F.avg("LONGITUDE").alias("avg_LONGITUDE"),
    F.avg("ELEVATION").alias("avg_ELEVATION"),
    F.avg("TEMP").alias("avg_TEMP"),
    F.avg("DEWP").alias("avg_DEWP"),
    F.avg("SLP").alias("avg_SLP"),
    F.avg("STP").alias("avg_STP"),
    F.avg("VISIB").alias("avg_VISIB"),
    F.avg("WDSP").alias("avg_WDSP"),
    F.avg("MXSPD").alias("avg_MXSPD"),
    F.avg("GUST").alias("avg_GUST"),
    F.avg("MAX").alias("avg_MAX"),
    F.avg("MIN").alias("avg_MIN"),
    F.avg("PRCP").alias("avg_PRCP"),
    F.avg("SNDP").alias("avg_SNDP"),
    F.avg("FRSHTT").alias("avg_FRSHTT")
)

# Debug future weekly averages
print("Future weekly averages:")
future_weekly_avg.show(5)

# Add Derived Features
print("Adding derived features for future data...")
future_weather = future_weekly_avg.withColumn("sin_week", F.sin(2 * F.lit(3.141592653589793) * col("Weather_Week") / 52)) \
                                  .withColumn("cos_week", F.cos(2 * F.lit(3.141592653589793) * col("Weather_Week") / 52)) \
                                  .withColumn("temp_prcp_interaction", col("avg_TEMP") * col("avg_PRCP"))

# Rename Year and Week Columns
future_weather = future_weather.withColumnRenamed("Weather_Year", "year") \
                               .withColumnRenamed("Weather_Week", "weekofyear")

# Add placeholder columns for lag features
future_weather = future_weather.withColumn("lag_cases", F.lit(0.0)) \
                               .withColumn("lag_cases_2", F.lit(0.0)) \
                               .withColumn("lag_cases_3", F.lit(0.0))
# Debug updated schema
print("Future weather data after renaming columns:")
future_weather.printSchema()
future_weather.show(5)

# Assemble Features
print("Assembling features for future predictions...")
future_feature_columns = ["year", "weekofyear", "avg_TEMP", "avg_PRCP", "avg_DEWP", "lag_cases", "lag_cases_2", "lag_cases_3", "sin_week", "cos_week", "temp_prcp_interaction"]
future_assembler = VectorAssembler(inputCols=future_feature_columns, outputCol="features")
future_features = future_assembler.transform(future_weather)

# Debugging feature assembly
print("Future features assembled:")
# Check the length of the feature vectors
feature_lengths = future_features.select("features").rdd.map(lambda row: len(row.features)).distinct().collect()
print(f"Distinct feature lengths in future features: {feature_lengths}")

print("Validating feature vector content...")

# Extract the 'values' field for validation
def extract_values(vector):
    if vector is not None:
        return vector.values.tolist()
    else:
        return []

# Register the UDF
extract_values_udf = F.udf(lambda x: extract_values(x), ArrayType(DoubleType()))

# Apply the UDF to extract the 'values' field
feature_validation = future_features.withColumn("feature_array", extract_values_udf(F.col("features")))

# Show the extracted feature array for debugging
feature_validation.select("feature_array").show(5, truncate=False)
# Make Predictions
print("Making predictions for future data...")
try:
    future_predictions = model.transform(future_features)
    future_predictions = future_predictions.withColumn("Prediction Cases", expm1(F.col("log_prediction")))

    # Show Predictions
    print("Future Predictions:")
    future_predictions.select("year", "weekofyear", "Prediction Cases").show()
    push_predictions_to_influxdb(future_predictions)
    
    # Debugging predictions
    print("Prediction Debug Info:")
    future_predictions.select("features", "log_prediction", "Prediction Cases").show(5)
except Exception as e:
    print(f"Error during prediction: {e}")

# Stop SparkSession
spark.stop()    