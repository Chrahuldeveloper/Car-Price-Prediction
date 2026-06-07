# Car Price Prediction Using Linear Regression From Scratch

## Overview

This project implements **Linear Regression from scratch using NumPy** to predict the selling price of used cars. The model is trained using **Gradient Descent** without relying on machine learning libraries such as Scikit-Learn's LinearRegression.

The dataset is sourced from CarDekho and contains information about various used cars, including manufacturing year, present price, kilometers driven, and transmission type.

## Features

* Linear Regression implemented from scratch
* Gradient Descent optimization
* Mean Squared Error (MSE) loss function
* R² Score evaluation
* Training loss visualization using Matplotlib
* Custom prediction function for new car data
* Dataset preprocessing and feature selection

## Dataset

Dataset: CarDekho Vehicle Dataset

Features used:

* Year
* Present_Price
* Kms_Driven
* Transmission

Target:

* Selling_Price

Dropped columns:

* Car_Name
* Seller_Type
* Owner
* Fuel_Type

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-Learn (only for train-test split)
* KaggleHub

## Project Structure

```text
.
├── main.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/car-price-prediction.git
cd car-price-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

```bash
python main.py
```

The script will:

1. Download the dataset.
2. Preprocess the data.
3. Train a Linear Regression model using Gradient Descent.
4. Display training and testing MSE.
5. Calculate R² scores.
6. Plot the training loss curve.
7. Predict the selling price for a sample car.

This project is open-source and available under the MIT License.
