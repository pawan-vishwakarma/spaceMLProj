import random
import string
import matplotlib.pyplot as plt

def generate_random_string(length=12):
    """
    Generate a random string of given length.
    """
    return ''.join(random.choices(string.ascii_letters, k=length))

def encode_string(s):
    """
    Encode string using ASCII values of characters.
    """
    return [ord(char) for char in s]

# Generate random original and predicted strings
original_string = generate_random_string()
predicted_string = generate_random_string()

# Encode original and predicted strings
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
