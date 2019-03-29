import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as lr
import matplotlib.pyplot as plt

data = pd.read_csv("dollar_sterling.csv")

x = data["day"]
y = data["rate"]

x = x.values.reshape(200, 1)
y = y.values.reshape(200, 1)

linearRegression = lr()

linearRegression.fit(x, y)

linearRegression.predict(x)

m = linearRegression.coef_
b = linearRegression.intercept_

a = np.arange(150)

plt.scatter(x, y)
plt.scatter(a, m*a+b, c="red", marker=">")
plt.show()

z = int(input("Which Day?"))
guess = m*z+b
print("Guess => ", guess)

plt.scatter(z, guess, c="blue", marker=">")
plt.show()
print("y = ", m, "x + ", b)