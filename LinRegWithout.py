import matplotlib.pyplot as plt
import math
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 6]

n = len(x)

sum_x = sum(x)
sum_y = sum(y)
sum_xy = sum(x[i] * y[i] for i in range(n))
sum_x2 = sum(x[i] ** 2 for i in range(n))

m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
c = (sum_y - m * sum_x) / n

print("Slope (m):", m)
print("Intercept (c):", c)

def predict(x_val):
    return m * x_val + c

y_pred = [predict(xi) for xi in x]
mse = sum((y[i] - y_pred[i])**2 for i in range(n)) / n
rmse = math.sqrt(mse)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

x_test = 6
y_test = predict(x_test)

plt.figure(figsize=(8, 5))

plt.scatter(x, y, label="Actual Data", marker="o")
plt.plot(x, y_pred, label="Regression Line")
plt.scatter(x_test, y_test, label="Prediction Point (x=6)", marker="*")

plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Linear Regression from Scratch")

plt.legend()
plt.grid(True)
plt.show()