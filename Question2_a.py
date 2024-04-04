import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Provided data
random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

# Calculate sigmoid values for each data point
sigmoid_values = [sigmoid(value) for value in random_values]

# Plot sigmoid function
x = np.linspace(-5, 5, 100)
y = sigmoid(x)

plt.figure(figsize=(10, 5))
plt.plot(x, y, label='Sigmoid')
plt.scatter(random_values, sigmoid_values, color='red', label='Data Points')
plt.title('Sigmoid Function and Provided Data Points')
plt.xlabel('x')
plt.ylabel('Sigmoid(x)')
plt.legend()
plt.grid(True)
plt.show()

# Print sigmoid values for provided data
print("Sigmoid Values:")
for i, value in enumerate(random_values):
    sigmoid_value = sigmoid_values[i]
    print(f"Sigmoid({value}) = {sigmoid_value}")
