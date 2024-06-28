# House Price Prediction

This repository contains a machine learning project aimed at predicting house prices using various regression models. The project includes data exploration, feature engineering, model building, evaluation, and testing. The notebooks used in this project are `training.ipynb` for training the model and `testing.ipynb` for evaluating the model's performance on new data.

## Table of Contents

- [Introduction](#introduction)
- [Data Description](#data-description)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Model Testing](#model-testing)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The goal of this project is to develop a model that can accurately predict house prices based on various features. The project employs multiple machine learning algorithms and selects the best-performing model based on evaluation metrics.

## Data Description

The dataset used in this project includes various features related to houses such as:
- `PRODUCT_ID`
- `PRODUCT_NAME`
- `PRODUCT_CONDITION`
- `CATEGORY`
- `PRODUCT_BRAND`
- `SHIPPING_AVAILABILITY`
- `PRODUCT_DESCRIPTION`

## Installation

To run this project, you need to have Python and the following libraries installed:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install the required libraries using pip:
```sh
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. Clone this repository:
   ```sh
   git clone https://github.com/kushagraag/Snap_it-ML_House_prediction.git
    ```
2. Navigate to the project directory:
    ```sh
    cd Snap_it-ML_House_prediction
    ```
3. Run the Jupyter notebooks:
    ```sh
    jupyter notebook
    ```

## Project Structure

- `training.ipynb`: Jupyter notebook for data exploration, feature engineering, model training, and evaluation.
- `testing.ipynb`: Jupyter notebook for testing the trained model on new data.
- `README.md`: Project documentation.

## Model Training

The `training.ipynb` notebook includes the following steps:

1. **Imports and Data Loading**: Load necessary libraries and the dataset.
2. **Data Exploration and Cleaning**: Explore the dataset and handle missing values.
3. **Feature Engineering**: Create new features and preprocess existing ones.
4. **Model Building**: Train multiple regression models including Linear Regression, Decision Trees, and Random Forest.
5. **Model Evaluation**: Evaluate models using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.
6. **Hyperparameter Tuning**: Optimize model parameters using techniques like Grid Search.
7. **Final Model and Predictions**: Select the best model and make predictions.

## Model Testing

The `testing.ipynb` notebook includes:

1. **Imports and Data Loading**: Load necessary libraries and the test dataset.
2. **Data Exploration**: Inspect the test data and handle any missing values.
3. **Model Evaluation**: Evaluate the performance of the trained model on the test data.

## Results

The model performance is evaluated based on various metrics, and the results are visualized to provide insights into the model's accuracy and reliability. The best-performing model is selected based on these metrics.

## Contributing

Contributions are welcome!

## License

This project is licensed under the MIT License.
