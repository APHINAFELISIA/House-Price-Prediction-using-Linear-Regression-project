import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("Housing.csv")

X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]
y = df['price']

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "house_price_model.pkl")

print("Model saved successfully!")

prediction = model.predict(input_data)
# Load dataset
data = pd.read_csv("Housing.csv")

# Numeric features only
X = data[["area", "bedrooms", "bathrooms", "stories", "parking"]]
y = data["price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# Save model
joblib.dump(model, "house_price_model.pkl")
print("Model Saved Successfully!")

# Graph
plt.scatter(data["area"], data["price"])
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Area vs Price")
plt.show()