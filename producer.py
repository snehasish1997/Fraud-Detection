# src/kafka_producer.py
import json
import time
from kafka import KafkaProducer
import random
import yaml

# Load Kafka configuration
with open("src/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

producer = KafkaProducer(
    bootstrap_servers=config['kafka']['bootstrap_servers'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def generate_transaction():
    transaction = {
        "transaction_id": random.randint(1000, 100000),
        "user_id": random.randint(1, 1000),
        "transaction_amount": random.uniform(5, 1000),
        "transaction_location": f"{random.uniform(-90, 90)}, {random.uniform(-180, 180)}",
        "transaction_timestamp": time.time(),
        "device_info": random.choice(["iPhone", "Android", "Desktop"]),
    }
    return transaction

def produce_data():
    while True:
        transaction = generate_transaction()
        producer.send(config['kafka']['topic'], transaction)
        time.sleep(1)  # Adjust to control transaction frequency

if __name__ == "__main__":
    produce_data()
