from kafka import KafkaProducer
import subprocess
import argparse


def stream_to_kafka(file_path, kafka_topic):
    """
    Streams data from a specific HDFS file to a Kafka topic.

    Parameters:
        file_path (str): The path of the file in HDFS.
        kafka_topic (str): The Kafka topic to publish the data to.
    """
    producer = KafkaProducer(bootstrap_servers="localhost:9092")

    try:
        # Read the file content
        content = subprocess.check_output(
            f"hdfs dfs -cat '/climate/raw/{file_path}'", shell=True
            ).decode()

        # Stream each line to Kafka
        for line in content.splitlines():
            producer.send(kafka_topic, line.encode("utf-8"))
            print(f"Sent line to Kafka: {line}")

    except subprocess.CalledProcessError as e:
        print(f"Failed to read file {file_path}: {e}")

    print("Data streaming to Kafka completed.")
    producer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream a specific HDFS file to Kafka.")
    parser.add_argument(
        "--file", type=str, required=True, help="HDFS file path to stream data from"
    )
    parser.add_argument(
        "--topic", type=str, required=True, help="Kafka topic to send data to"
    )

    args = parser.parse_args()

    stream_to_kafka(args.file, args.topic)