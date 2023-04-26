from pyspark.sql.functions import col
import numpy as np

data = spark.read.option("inferSchema", "true").option("header", "true").csv("/FileStore/tables/SMS_train.csv")

def get_label(x):
    if x == "rain" or x == "drizzle" or x == "snow":
        return 1
    else:
        return 0

def init_params():
    W1 = np.random.rand(2, 4) - 0.5
    b1 = np.random.rand(2, 1) - 0.5
    W2 = np.random.rand(2, 2) - 0.5
    b2 = np.random.rand(2, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    m = X.shape[1]
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 100 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

data = data.rdd.map(lambda x: (x['date'], x['precipitation'], x['temp_max'], x['temp_min'], x['wind'], x['weather'], get_label(x['weather']))).toDF(["date", 'precipitation', "temp_max", "temp_min", "wind", "weather", "label"])

train = data.limit(1000)
test = data.subtract(train).orderBy("date")

X = np.array(train.select("precipitation", "temp_max", "temp_min", "wind").collect())
X = X.T
Y = np.array(train.select("label").collect()).flatten()
Y = Y.T

W1, b1, W2, b2 = gradient_descent(X, Y, 0.05, 1000)