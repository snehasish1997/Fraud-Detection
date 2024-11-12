import yaml
import stripe
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType

# Load configuration
with open("src/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Initialize Spark session
spark = SparkSession.builder \
    .appName(config['spark']['app_name']) \
    .master(config['spark']['master']) \
    .getOrCreate()

# Set Stripe API key
stripe.api_key = config['stripe']['secret_key']

def fetch_data_from_stripe():
    # Fetch transaction data from Stripe
    transactions = stripe.Charge.list(limit=100)  # Modify limit or use pagination as necessary
    
    # Process and extract relevant fields from Stripe API response
    records = []
    for charge in transactions.auto_paging_iter():
        records.append({
            'transaction_id': charge['id'],
            'user_id': charge['customer'] if charge['customer'] else "unknown",
            'transaction_timestamp': charge['created'],  # Unix timestamp
            'transaction_amount': charge['amount'] / 100.0,  # Convert from cents to dollars
            'currency': charge['currency'],
            'device_info': charge['payment_method_details']['type'] if 'payment_method_details' in charge else "unknown",
            'location_lat': None,  # Stripe API does not provide location; set as None or use placeholder
            'location_long': None  # Same as above
        })

    # Define schema for the DataFrame
    schema = StructType([
        StructField("transaction_id", StringType(), True),
        StructField("user_id", StringType(), True),
        StructField("transaction_timestamp", TimestampType(), True),
        StructField("transaction_amount", DoubleType(), True),
        StructField("currency", StringType(), True),
        StructField("device_info", StringType(), True),
        StructField("location_lat", DoubleType(), True),
        StructField("location_long", DoubleType(), True)
    ])

    # Create a Spark DataFrame from the list of records
    df = spark.createDataFrame(records, schema=schema)
    return df

def feature_engineering(df):
    window = Window.partitionBy("user_id").orderBy("transaction_timestamp")

    # Amount deviation from user's average transaction amount
    df = df.withColumn("user_avg_transaction", F.avg("transaction_amount").over(window))
    df = df.withColumn("amount_deviation", F.abs(df["transaction_amount"] - df["user_avg_transaction"]))

    # Transaction velocity (frequency of transactions within a short time frame)
    df = df.withColumn("time_diff", F.col("transaction_timestamp") - F.lag("transaction_timestamp", 1).over(window))
    df = df.withColumn("velocity", F.when(F.col("time_diff").isNotNull(), 1 / F.col("time_diff").cast("double")).otherwise(0))

    # Device change indicator
    df = df.withColumn("device_change", (F.col("device_info") != F.lag("device_info", 1).over(window)).cast("int"))

    # Time of day as a feature
    df = df.withColumn("time_of_day", F.hour(F.from_unixtime("transaction_timestamp")))

    return df

def preprocess_and_save():
    df = fetch_data_from_stripe()
    df = feature_engineering(df)
    df.write.parquet(config['data_path'], mode="overwrite")

if __name__ == "__main__":
    preprocess_and_save()
