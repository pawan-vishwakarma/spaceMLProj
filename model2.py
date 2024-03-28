# Model to correct the errors in a string with 8-bit binary representation of characters while transfering data over a network in space.
import random
import string
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 

  
def encode_string(s):
    """
    Encode string using ASCII values of characters.
    """
    return [ord(char) for char in s]


def generate_random_string(length=12):
    """
    Generate a random string of given length.
    """
    return ''.join(random.choices(string.ascii_letters, k=length))


# Data preparation
X = []
y = []
for character in range(48, 58):
  y.append(chr(character))
  X.append(list(bin(character)[2:].zfill(8)))

for character in range(65, 91):
  y.append(chr(character))
  X.append(list(bin(character)[2:].zfill(8)))

for character in range(97, 123):
  y.append(chr(character))
  X.append(list(bin(character)[2:].zfill(8)))

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X, y)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the model: ",accuracy)
print("----------------------------------------------")


# Predicting a string with errors in 8-bit binary representation
count = 0 
while count < 15:
  new_str = generate_random_string()
  predicted_str = []
  num_of_error = 5
  idx_to_insert_erros = np.random.randint(0, len(new_str), num_of_error)
  for i in range(len(new_str)):
    temp = list(bin(ord(new_str[i]))[2:].zfill(8))
    if i in idx_to_insert_erros:
      temp[np.random.randint(0, 8)] = str(np.random.randint(0, 2))
    predicted_str.append(model.predict([temp])[0])

  print("Original String:", new_str)
  print("Predicted String:", "".join(predicted_str))
  print("------------------------------------------")
  count += 1

#Plotting graph
original_string = generate_random_string()
predicted_string = generate_random_string()

original_encoded = encode_string(original_string)
predicted_encoded = encode_string(predicted_string)

# Plot original and predicted strings
plt.plot(original_encoded, label='Original')
plt.plot(predicted_encoded, label='Predicted')
plt.xlabel('Character Index')
plt.ylabel('ASCII Value')
plt.title('Original vs Predicted Strings')
plt.legend()
plt.show()