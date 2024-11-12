import yaml
import stripe
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Load configuration
with open("src/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Initialize Spark session
spark = SparkSession.builder \
    .appName(config['spark']['app_name']) \
    .master(config['spark']['master']) \
    .getOrCreate()

# Initialize Stripe with the API key from config
stripe.api_key = config['stripe']['secret_key']

def load_data():
    # Fetch transaction data from Stripe
    transactions = stripe.Charge.list(limit=100)

    # Extract relevant fields
    records = []
    for charge in transactions.auto_paging_iter():
        records.append({
            'transaction_id': charge['id'],
            'amount': charge['amount'] / 100.0,  # Stripe returns amount in cents, convert to dollars
            'currency': charge['currency'],
            'status': charge['status'],
            'is_fraud': 1 if charge['status'] == 'failed' else 0  # Example logic for fraud
        })
    
    # Check if records are empty and handle the case
    if not records:
        raise ValueError("No transactions found in Stripe. Ensure there is data in your account.")

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
