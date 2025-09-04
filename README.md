Linear Regression Tricks and Gradient Descent

This project demonstrates different approaches to linear regression using custom update "tricks" (simple, square, absolute) and compares them with gradient descent optimization. It also includes visualization and error evaluation with RMSE.

🚀 Features

Implements three heuristic tricks for updating regression parameters:

Simple Trick – adjusts weight and bias based on the relative position of prediction and actual value.

Square Trick – updates using squared error (closer to gradient descent).

Absolute Trick – updates using absolute error difference.

Implements Gradient Descent for linear regression with Mean Squared Error (MSE).

Includes loss functions:

Square Loss

Absolute Loss

RMSE

Visualization of results for all methods.

📂 Project Structure
.
├── linear_regression_tricks.py   # main script with all methods
├── README.md                     # project documentation

⚙️ How It Works

The dataset is defined manually:

features = np.array([1, 2, 3, 4, 5, 6, 7])
labels   = np.array([155, 197, 244, 300, 356, 407, 448])


Different tricks are applied via the linearRegression() function.

The model is further optimized using gradient descent.

RMSE is calculated for performance comparison.

Results are visualized with Matplotlib.

📊 Example Output

Console output:

Simple Trick: y = 63.77 x + 99.33 | RMSE = 10.23
Square Trick: y = 63.95 x + 92.11 | RMSE = 9.87
Absolute Trick: y = 64.10 x + 95.54 | RMSE = 11.02
Final Model: y = 64.00 x + 91.50
RMSE (square trick with gradient descent) = 8.75


Visualization:

Square Trick

Simple Trick

Absolute Trick

Square Trick with Gradient Descent

Each subplot shows the dataset points and the fitted regression line.
