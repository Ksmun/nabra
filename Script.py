import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from datetime import datetime



dates = [] # To save the dates
prices = [] # To save the prices


def get_data(filename):
    with open(filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip header row
        for row in csv_reader:
            # Convert date string (e.g., "1999-11-18") to an ordinal number
            date_obj = datetime.strptime(row[0], '%Y-%m-%d')
            dates.append(date_obj.toordinal())
            # Use the closing price (column index 4)
            prices.append(float(row[4]))
    return


def predict_price(dates, prices, x):
    # Reshape dates into a matrix as required by scikit-learn
    dates_np = np.reshape(dates, (len(dates), 1))

    # Define SVR models with different kernels
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    # Fit the models to the data
    svr_lin.fit(dates_np, prices)
    svr_poly.fit(dates_np, prices)
    svr_rbf.fit(dates_np, prices)

    # Create a new figure for plotting
    plt.figure(figsize=(12, 8))

    # Plot the original data
    plt.scatter(dates_np, prices, color='black', label='Data')

    # Plot predictions of each model over the range of dates
    plt.plot(dates_np, svr_rbf.predict(dates_np), color='red', label='RBF model')
    plt.plot(dates_np, svr_lin.predict(dates_np), color='green', label='Linear model')
    plt.plot(dates_np, svr_poly.predict(dates_np), color='blue', label='Polynomial model')

    # Predict the price for the future date 'x'
    x_val = np.array([[x]])
    future_rbf = svr_rbf.predict(x_val)[0]
    future_lin = svr_lin.predict(x_val)[0]
    future_poly = svr_poly.predict(x_val)[0]

    # Plot the predicted future point for each model
    plt.scatter(x, future_rbf, color='red', marker='o', s=100, label='Future Prediction RBF')
    plt.scatter(x, future_lin, color='green', marker='o', s=100, label='Future Prediction Linear')
    plt.scatter(x, future_poly, color='blue', marker='o', s=100, label='Future Prediction Poly')

    # Annotate the predicted points with their price values
    plt.annotate(f"{future_rbf:.2f}", (x, future_rbf), textcoords="offset points", xytext=(0, 10), ha='center',
                 color='red')
    plt.annotate(f"{future_lin:.2f}", (x, future_lin), textcoords="offset points", xytext=(0, 10), ha='center',
                 color='green')
    plt.annotate(f"{future_poly:.2f}", (x, future_poly), textcoords="offset points", xytext=(0, 10), ha='center',
                 color='blue')

    # Labeling the plot
    plt.xlabel('Date (Ordinal)')
    plt.ylabel('Closing Price')
    plt.title('SVR: Stock Price Prediction')
    plt.legend()

    # Save and show the plot
    plt.savefig('output/svr_stock_prediction.png')
    plt.show()

    return future_rbf, future_lin, future_poly


# Load your dataset from the CSV file (update the file path if necessary)
get_data('WOR.csv')

# For example, predict the closing price for the future date '2000-01-03'
future_date = datetime.strptime('2000-01-03', '%Y-%m-%d').toordinal()
predicted_price = predict_price(dates, prices, future_date)

print("Predicted Closing Price for 2000-01-03:")
print("RBF Kernel:", predicted_price[0])
print("Linear Kernel:", predicted_price[1])
print("Polynomial Kernel:", predicted_price[2])