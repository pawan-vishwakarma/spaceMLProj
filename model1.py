# Model to predict the resolution time of an incident based on the priority, category, and the hour of the day the incident was reported in the IT service desk.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


np.random.seed(42)
num_samples = 10000

# Incident priority (1: Low, 2: Medium, 3: High)
priority = np.random.choice([1, 2, 3], size=num_samples)

# Incident category (1: Hardware, 2: Software, 3: Network)
category = np.random.choice([1, 2, 3], size=num_samples)

# Hour of the day the incident was reported (0-23)
reported_hour = np.random.randint(0, 24, size=num_samples)

noise = np.random.normal(loc=0, scale=5, size=num_samples)
priority_coeff = 5
category_coeff = 10
hour_coeff = 2
resolution_time = priority * priority_coeff + category * category_coeff + reported_hour * hour_coeff + noise

incident_data = pd.DataFrame({
    'Priority': priority,
    'Category': category,
    'Reported_Hour': reported_hour,
    'Resolution_Time': resolution_time
})

X = incident_data[['Priority', 'Category', 'Reported_Hour']].values
y = incident_data['Resolution_Time'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
idx = 0 
for i in range(len(y_test)):
  print(f"Actual resolved time: {y_test[i]}, Predicted resolve time: {y_pred[i]}")
  if abs(y_test[i] - y_pred[i]) < 10:
    idx += 1
print(f"Accuracy with [plus/minus 10]: {idx/len(y_test)}")
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)


# Plot the graph
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Resolution Time")
plt.ylabel("Predicted Resolution Time")
plt.title("Actual vs Predicted Resolution Time")
plt.show()