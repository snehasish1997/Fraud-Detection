
# Fraud Detection System

This repository contains a real-time fraud detection system designed to identify and flag high-risk transactions. Using Spark and a Random Forest model, this project aims to deliver accurate, efficient fraud detection for financial applications.

## Project Structure

- **a.py**: Entry point for the application. This script initializes the main processes.
- **config.yaml**: Configuration file with parameters like model paths, thresholds, and data source settings.
- **model_train.py**: Script to train the primary fraud detection model.
- **model2.py**: An alternate model script for experimentation or model comparison.
- **model3.py**: Additional model script, potentially for ensemble methods or testing variations.
- **preprocess.py**: Preprocessing pipeline to clean and prepare data for model training.
- **producer.py**: Manages data ingestion, simulating real-time data streaming for testing the system.

## Features

- **Real-Time Detection**: Identifies fraudulent transactions as they occur.
- **High Accuracy**: Uses advanced feature engineering and a Random Forest model for high precision.
- **Scalable Processing**: Built with Spark to handle large transaction volumes.

## Installation

To get started, clone the repository and install the dependencies.

```bash
git clone https://github.com/yourusername/fraud-detection-system.git
cd fraud-detection-system
pip install -r requirements.txt
```

## Configuration

Adjust settings in `config.yaml` as needed:
- **model_path**: Path to the trained model file.
- **threshold**: Probability threshold for classifying transactions as fraudulent.
- **data_source**: Path to or specifications for the data source.

## Usage

1. **Preprocess Data**:
   Run the data preprocessing pipeline:
   ```bash
   python preprocess.py
   ```

2. **Train Model**:
   Train the fraud detection model with:
   ```bash
   python model_train.py
   ```

3. **Run Fraud Detection**:
   To start real-time fraud detection, execute:
   ```bash
   python a.py
   ```

## Dependencies

The project relies on the following Python libraries:
- Spark
- Scikit-Learn
- Pandas
- PyYAML

## Contributing

Contributions are welcome! Feel free to submit pull requests with improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
