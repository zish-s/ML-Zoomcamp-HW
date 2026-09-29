import pandas as pd
import numpy as np

df = pd.read_csv("car_fuel_efficiency_2026.csv")


#Q1
print("Q1. Pandas version:")
print(pd.__version__)


#Q2
print("Q2:")

print(len(df))


# Q3
print("Q3:")

print(df["fuel_type"].nunique())


# Q4
print("Q4:")

print(df.isnull().any().sum())


# Q5
print("Q5:")

asia = df[df["origin"] == "Asia"]
print(asia["fuel_efficiency_mpg"].max())


# Q6
print("Q6 (median):")


horsepower = df["horsepower"]

median_before = horsepower.median()
most_frequent = horsepower.mode()[0]

df["horsepower"] = df["horsepower"].fillna(most_frequent)

median_after = df["horsepower"].median()

print("Before:", median_before)
print("After:", median_after)


# Q7

asia = df[df["origin"] == "Asia"]

X = asia[["vehicle_weight", "model_year"]].head(7).to_numpy()

XTX = X.T @ X

XTX_inv = np.linalg.inv(XTX)

y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

# Calculate weights
w = XTX_inv @ X.T @ y

print("\nQ7. Weights:")
print(w)

print("Sum of weights:")
print(w.sum())