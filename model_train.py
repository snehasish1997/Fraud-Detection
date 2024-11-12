# src/model_train_spark.py
import yaml
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import requests

# Load configuration
with open("src/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Initialize Spark session
spark = SparkSession.builder \
    .appName(config['spark']['app_name']) \
    .master(config['spark']['master']) \
    .getOrCreate()

#def load_data():
#    return spark.read.parquet("data/processed_transactions")


def load_data():
    # PayPal API credentials
    client_id = config['paypal']['client_id']
    client_secret = config['paypal']['client_secret']
    
    # Get access token
    auth_response = requests.post(
        'https://api.sandbox.paypal.com/v1/oauth2/token',
        #'https://sandbox.paypal.com',
        headers={'Accept': 'application/json', 'Accept-Language': 'en_US'},
        data={'grant_type': 'client_credentials'},
        auth=(client_id, client_secret)
    )
    
    if auth_response.status_code != 200:
        raise Exception("Failed to authenticate with PayPal API: " + auth_response.text)
    
    access_token = auth_response.json()['access_token']
    
    # Fetch transaction data (example endpoint)
    transactions_response = requests.get(
        'https://api.sandbox.paypal.com/v1/reporting/transactions',
        #'https://sandbox.paypal.com',
        headers={'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'},
        params={'start_date': '2023-01-01T00:00:00Z', 'end_date': '2023-12-31T23:59:59Z', 'fields': 'all'}
    )
    
    if transactions_response.status_code != 200:
        raise Exception("Failed to fetch transaction data: " + transactions_response.text)
    
    transactions_data = transactions_response.json()
    
    # Convert transaction data into a format suitable for DataFrame
    # This depends on the structure of the data returned by the PayPal API
    transaction_list = transactions_data.get('transaction_details', [])
    
    # Extract relevant fields
    records = []
    for transaction in transaction_list:
        records.append({
            'transaction_id': transaction['transaction_info']['transaction_id'],
            'amount': float(transaction['transaction_info']['transaction_amount']['value']),
            'currency': transaction['transaction_info']['transaction_amount']['currency'],
            'status': transaction['transaction_info']['transaction_status'],
            'is_fraud': 1 if transaction['transaction_info']['transaction_status'] == 'FAILED' else 0  # Example logic for fraud
        })
    
    # Create a Spark DataFrame from the records
    df = spark.createDataFrame(records)
    
    return df




def train_model():
    df = load_data()

    # Assemble features into a single vector
    assembler = VectorAssembler(inputCols=config['features'], outputCol="features")
    df = assembler.transform(df)

    # Split the data
    train, test = df.randomSplit([0.7, 0.3], seed=42)

    # Model initialization and training
    rf = RandomForestClassifier(featuresCol="features", labelCol="is_fraud", numTrees=100)
    model = rf.fit(train)

    # Evaluation
    predictions = model.transform(test)
    evaluator = BinaryClassificationEvaluator(labelCol="is_fraud", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
    print(f"Test AUC: {auc}")

    # Save the model
    model.save(config['model_path'])

if __name__ == "__main__":
    train_model()
