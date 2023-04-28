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
    return W1, b1, W2

def softmax(Z):
    f = np.exp(Z - np.max(Z))  # shift values
    return f / f.sum(axis=0)
    
def forward_prop(W1, b1, W2, x1, x2):
    A1 = W1.dot(x1) + b1
    O1 = softmax(A1)
    A2 = W1.dot(x2) + b1 + W2.dot(A1)
    O2 = softmax(A2)
    return A1, A2, O1, O2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(A1, A2, O2, W1, W2, x1, x2, y2):
    y = one_hot(y2)
    m = 1000
    dA2 = O2 - y
    dW2 = dA2.dot(A1.T) / m
    dW1 = (dA2.dot(x2.T) + W2.T.dot(dA2).dot(x1.T)) / 1000
    db1 = (dA2 + W2.T.dot(dA2)) / 1000
    return dW1, db1, dW2

def update_params(W1, b1, W2, dW1, db1, dW2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2     
    return W1, b1, W2

def get_predictions(pred):
    return np.argmax(pred, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2 = init_params()
    for i in range(iterations):
        pred_probs = np.zeros((2, 1000))
        for j in range(X.shape[1]-1):
            x1, x2 = np.reshape(X[:, j], (4,1)), np.reshape(X[:, j+1], (4,1))
            A1, A2, O1, O2 = forward_prop(W1, b1, W2, x1, x2)
            dW1, db1, dW2 = backward_prop(A1, A2, O2, W1, W2, x1, x2, Y[j+1])
            W1, b1, W2 = update_params(W1, b1, W2, dW1, db1, dW2, alpha)
            if j == 0:
                pred_probs[0, 0] = O1[0, 0]
                pred_probs[1, 0] = O1[1, 0]
            pred_probs[0, j+1] = O2[0, 0]
            pred_probs[1, j+1] = O2[1, 0]
        if i % 50 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(pred_probs)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2

def predict_test(X, W1, b1, W2):
    pred_probs = np.zeros((2, X.shape[1]))
    for j in range(X.shape[1]-1):
        x1, x2 = np.reshape(X[:, j], (4,1)), np.reshape(X[:, j+1], (4,1))
        A1, A2, O1, O2 = forward_prop(W1, b1, W2, x1, x2)
        if j == 0:
            pred_probs[0, 0] = O1[0, 0]
            pred_probs[1, 0] = O1[1, 0]
        pred_probs[0, j+1] = O2[0, 0]
        pred_probs[1, j+1] = O2[1, 0]
    predictions = get_predictions(pred_probs)
    return predictions

data = data.rdd.map(lambda x: (x['date'], x['precipitation'], x['temp_max'], x['temp_min'], x['wind'], x['weather'], get_label(x['weather']))).toDF(["date", 'precipitation', "temp_max", "temp_min", "wind", "weather", "label"])

train = data.limit(1000)
test = data.subtract(train).orderBy("date")

X = np.array(train.select("precipitation", "temp_max", "temp_min", "wind").collect())
X = X.T
Y = np.array(train.select("label").collect()).flatten()
Y = Y.T

W1, b1, W2 = gradient_descent(X, Y, 0.005, 200)

X_test = np.array(test.select("precipitation", "temp_max", "temp_min", "wind").collect())
X_test = X_test.T
Y_test = np.array(test.select("label").collect()).flatten()
Y_test = Y_test.T

test_pred = predict_test(X_test, W1, b1, W2)
print(get_accuracy(test_pred, Y_test))