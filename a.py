# src/api.py
from flask import Flask, request, jsonify
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
import yaml

app = Flask(__name__)

# Load configuration and Spark model
with open("src/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

spark = SparkSession.builder.appName("FraudDetectionAPI").getOrCreate()
model = RandomForestClassificationModel.load(config['model_path'])

assembler = VectorAssembler(inputCols=config['features'], outputCol="features")

@app.route('/predict', methods=['POST'])
def fraud_detection():
    data = request.get_json()
    df = spark.createDataFrame([data])

    # Feature engineering and prediction
    df = assembler.transform(df)
    prediction = model.transform(df).collect()[0]

    return jsonify(fraud=bool(prediction['prediction']), confidence_score=prediction['probability'][1])

if __name__ == "__main__":
    app.run(debug=True)
