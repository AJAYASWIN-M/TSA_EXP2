### DEVELOPED BY: AJAY ASWIN M
### REGISTER NO: 212222240005

# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date:
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
 Import necessary libraries (NumPy, Matplotlib)
 Load the dataset
 Calculate the linear trend values using least square method
 Calculate the polynomial trend values using least square method
 End the program
 
### PROGRAM:
A - LINEAR TREND ESTIMATION
```python
# LINEAR TREND ESTIMATION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('/content/petrol.csv')
data['Date'] = pd.to_datetime(data['Date'])
price = data.groupby('Date')['Delhi'].mean().reset_index()

# Linear trend estimation
x = np.arange(len(price))
y = price['Delhi']
linear_coeffs = np.polyfit(x, y, 1)
linear_trend = np.polyval(linear_coeffs, x)

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(price['Date'], price['Delhi'], label='Original Data', marker='o')
plt.plot(price['Date'], linear_trend, label='Linear Trend', color='red')
plt.title('Linear Trend Estimation')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

```

B- POLYNOMIAL TREND ESTIMATION
```python
# POLYNOMIAL TREND ESTIMATION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('/content/petrol.csv',nrows=50)
data['Date'] = pd.to_datetime(data['Date'])
price = data.groupby('Date')['Chennai'].mean().reset_index()

# Polynomial trend estimation (degree 2)
x = np.arange(len(price))
y = price['Chennai']
poly_coeffs = np.polyfit(x, y, 2)
poly_trend = np.polyval(poly_coeffs, x)

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(price['Date'], price['Chennai'], label='Original Data', marker='o')
plt.plot(price['Date'], poly_trend, label='Polynomial Trend (Degree 2)', color='green')
plt.title('Polynomial Trend Estimation')
plt.xlabel('Date')
plt.ylabel('price')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

```

### OUTPUT
A - LINEAR TREND ESTIMATION
![linear](https://github.com/user-attachments/assets/8aa590b1-5251-4a22-915d-dc4906277b99)


B- POLYNOMIAL TREND ESTIMATION
![polynomial](https://github.com/user-attachments/assets/c8ebf258-8c8f-4a05-9190-b76cbd5ea5a5)

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
