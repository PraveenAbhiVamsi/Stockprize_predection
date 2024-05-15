
# Stock Price Prediction using LSTM

This project utilizes Long Short-Term Memory (LSTM) neural networks to predict the future stock prices of Tesla (TSLA). The LSTM model is trained on historical stock price data and evaluated on a test dataset to assess its performance in predicting stock prices.

## Dataset
The dataset used for this project contains historical stock price data for Tesla (TSLA). The dataset includes features such as the date, closing price, volume, and other relevant attributes. The dataset is preprocessed and normalized before training the LSTM model.

## Requirements
- Python 3.x
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-price-prediction.git
   cd stock-price-prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset (Tesla stock price data) and place it in the project directory.

## Usage
1. Preprocess the data:
   - Run `preprocess_data.py` to load and preprocess the dataset. This script normalizes the data and prepares it for training the LSTM model.

2. Train the LSTM model:
   - Execute `train_model.py` to build and train the LSTM model using the preprocessed data. The model architecture, such as the number of LSTM units and epochs, can be adjusted in this script.

3. Evaluate the model:
   - After training, the model's performance can be evaluated using `evaluate_model.py`. This script calculates the training and testing losses and visualizes the actual vs. predicted stock prices.

## Results
The results of the stock price prediction are displayed visually through plots comparing the actual stock prices with the predicted prices. Additionally, the training and testing losses are printed to assess the model's performance.

## Acknowledgments

Special thanks to [**Leela**]([https://github.com/leela](https://github.com/LEELAPRIYA)) for their valuable contributions and partnership in this project. Their insights and efforts have significantly enriched the development process and outcomes.

## Contributing
Contributions to this project are welcome. Feel free to open issues or submit pull requests with improvements, bug fixes, or additional features.
