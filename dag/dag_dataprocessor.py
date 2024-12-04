from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta

# Default arguments for the DAG
default_args = {
    "owner": "kafka_receiver",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# Define the DAG
with DAG(
    "dsproject_kafka_receiver_pipeline",
    default_args=default_args,
    description="Kafka Receiver DAG using Spark to process messages",
    schedule_interval=None,
    start_date=datetime(2024, 12, 1),
    catchup=False,
) as dag:

    # Task: Submit Spark job for Kafka receiver
    kafka_receiver_task = SparkSubmitOperator(
        task_id="kafka_receiver",
        application="kafka_reciever.py",  # Path to your Spark receiver script
        packages="org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3",
        jars="spark-kafka-app-1.0-SNAPSHOT.jar",
        conf={"spark.executor.memory": "2g", "spark.driver.memory": "1g"},
        total_executor_cores=2,
        verbose=True,
    )