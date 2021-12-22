import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
#%matplotlib inline
from sklearn import preprocessing

def simpleLinearReg():
    df = pd.read_csv("/Users/anipandey/Documents/ML/coursera/FuelConsumptionCo2.csv")

    # take a look at the dataset
    df.head()
    # summarize the data
    df.describe()
    cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
    cdf.head(10)

    #Lets split our dataset into train and test sets, 80% of the entire data for training, and the 20% for testing
    msk = np.random.rand(len(df)) < 0.8
    train = cdf[msk]
    test = cdf[~msk]

    #Using sklearn package to model data.
    from sklearn import linear_model
    regr = linear_model.LinearRegression()

    #simple reg has only a single independent variable...
    train_x = np.asanyarray(train[['ENGINESIZE']])
    train_y = np.asanyarray(train[['CO2EMISSIONS']])
    regr.fit (train_x, train_y)
    # The coefficients
    print ('Coefficients: ', regr.coef_)
    print ('Intercept: ',regr.intercept_)

    #We can plot the fit line over the data
    plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
    plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.show()

    #lets use MSE here to calculate the accuracy of our model based on the test set
    from sklearn.metrics import r2_score

    test_x = np.asanyarray(test[['ENGINESIZE']])
    test_y = np.asanyarray(test[['CO2EMISSIONS']])
    test_y_ = regr.predict(test_x)

    print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
    print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
    print("R2-score: %.2f" % r2_score(test_y , test_y_) )

def multipleLinearReg():
    df = pd.read_csv("/Users/anipandey/Documents/ML/coursera/FuelConsumptionCo2.csv")
    # take a look at the dataset
    df.head()
    
    cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
    cdf.head(9)

    msk = np.random.rand(len(df)) < 0.8
    train = cdf[msk]
    test = cdf[~msk]

    plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.show()

    from sklearn import linear_model
    regr = linear_model.LinearRegression()

    #multiple reg has more then 1 independent variables...
    x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
    y = np.asanyarray(train[['CO2EMISSIONS']])
    regr.fit (x, y)
    # The coefficients
    print ('Coefficients: ', regr.coef_)

    #Ordinary Least Squares (OLS)
    #can find the best parameters using of the following methods:
    #Solving the model parameters analytically using closed-form equations
    # Using an optimization algorithm (Gradient Descent, Stochastic Gradient Descent, Newton’s Method, etc.)
    
    #prediction
    y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
    x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
    y = np.asanyarray(test[['CO2EMISSIONS']])
    print("Residual sum of squares: %.2f"
        % np.mean((y_hat - y) ** 2))

    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(x, y))

def polynomialReg():
    df = pd.read_csv("/Users/anipandey/Documents/ML/coursera/FuelConsumptionCo2_nonL.csv")
    # take a look at the dataset
    df.head()
    cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
    cdf.head(9)

    msk = np.random.rand(len(df)) < 0.8
    train = cdf[msk]
    test = cdf[~msk]

    #in scikit if we select the degree of the polynomial to be 2, then it generates 3 features, degree=0, degree=1 and degree=2
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn import linear_model
    train_x = np.asanyarray(train[['ENGINESIZE']])
    train_y = np.asanyarray(train[['CO2EMISSIONS']])

    test_x = np.asanyarray(test[['ENGINESIZE']])
    test_y = np.asanyarray(test[['CO2EMISSIONS']])


    poly = PolynomialFeatures(degree=2)
    train_x_poly = poly.fit_transform(train_x)
    
    #polynomial regression is considered to be a special case of traditional multiple linear regression
    clf = linear_model.LinearRegression()
    train_y_ = clf.fit(train_x_poly, train_y)
    # The coefficients
    print ('Coefficients: ', clf.coef_)
    print ('Intercept: ',clf.intercept_)

    plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
    XX = np.arange(0.0, 10.0, 0.1)
    yy = clf.intercept_[0]+ clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2)
    plt.plot(XX, yy, '-r' )
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.show()

    from sklearn.metrics import r2_score

    test_x_poly = poly.fit_transform(test_x)
    test_y_ = clf.predict(test_x_poly)

    print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
    print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
    print("R2-score: %.2f" % r2_score(test_y,test_y_ ) )

def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y

def nonLReg():
    df = pd.read_csv("/Users/anipandey/Documents/ML/coursera/china_gdp.csv")
    df.head(10)
    plt.figure(figsize=(8,5))
    x_data, y_data = (df["Year"].values, df["Value"].values)
    plt.plot(x_data, y_data, 'ro')
    plt.ylabel('GDP')
    plt.xlabel('Year')
    

    #hit ans trial fits well!
    beta_1 = 0.10 #0.20 Controls the curve's steepness
    beta_2 = 1990.0 #2012 Slides the curve on the x-axis

    #logistic function
    Y_pred = sigmoid(x_data, beta_1 , beta_2)

    #plot initial prediction against datapoints
    plt.plot(x_data, Y_pred*15000000000000.)
    plt.plot(x_data, y_data, 'ro')
    #plt.show()

    # Lets normalize our data
    xdata =x_data/max(x_data)
    ydata =y_data/max(y_data)

    #curve_fit which uses non-linear least squares to fit our sigmoid function, to data
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(sigmoid, xdata, ydata)
    #print the final parameters
    print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

    #Now we plot our resulting regression model.
    x = np.linspace(1960, 2015, 55)
    x = x/max(x)
    plt.figure(figsize=(8,5))
    y = sigmoid(x, *popt)
    plt.plot(xdata, ydata, 'ro', label='data')
    plt.plot(x,y, linewidth=3.0, label='fit')
    plt.legend(loc='best')
    plt.ylabel('GDP')
    plt.xlabel('Year')
    plt.show()

def logisticGraph():# alsp called sigmoid
        X = np.arange(-5.0, 5.0, 0.1)
        Y = 1.0 / (1.0 + np.exp(-X))

        plt.plot(X,Y) 
        plt.ylabel('Dependent Variable')
        plt.xlabel('Independent Variable')
        plt.show()


#simpleLinearReg()
#multipleLinearReg()
#polynomialReg()
#logisticGraph()
nonLReg()

