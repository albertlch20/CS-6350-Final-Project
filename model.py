from pyspark.sql.functions import col
import numpy as np

def get_label(x):
    if x == "rain" or x == "drizzle" or x == "snow":
        return 1
    else:
        return 0

def get_init_params():
    # Set initial weights to be -0.1 ~ 0.1
    W1 = np.random.uniform(low=-0.1, high=0.1, size=(2, 4))
    b1 = np.random.uniform(low=-0.1, high=0.1, size=(2, 1))
    W2 = np.random.uniform(low=-0.1, high=0.1, size=(2, 2))
    return W1, b1, W2

def softmax(Z):
    # Shift values to avoid overflow
    f = np.exp(Z - np.max(Z))  
    return f / f.sum(axis=0)
    
def forward(W1, b1, W2, x1, x2):
    A1 = W1.dot(x1) + b1
    O1 = softmax(A1)
    # Depends on previous result
    A2 = W1.dot(x2) + b1 + W2.dot(A1)
    O2 = softmax(A2)
    return A1, A2, O1, O2

def one_hot(y2):
    # Convert label to one-hot encoding, ex: 1 -> [0
    #                                              1]
    y = np.zeros((2, 1))
    y[y2, 0] = 1
    return y

def backward(A1, A2, O2, W1, W2, x1, x2, y2, m):
    # Backward propogation in RNN
    y = one_hot(y2)
    dA2 = O2 - y
    dW2 = dA2.dot(A1.T) / m
    dW1 = (dA2.dot(x2.T) + W2.T.dot(dA2).dot(x1.T)) / m
    db1 = (dA2 + W2.T.dot(dA2)) / m
    return dW1, db1, dW2

def update_params(W1, b1, W2, dW1, db1, dW2, alpha):
    return W1 - alpha * dW1, b1 - alpha * db1, W2 - alpha * dW2 

def predict(probs):
    # Convert probabilities to predicted labels
    return np.argmax(probs, 0)

def get_accuracy(preds, Y):
    return np.sum(preds == Y) / Y.size

def train(X, Y, alpha, iterations):
    # The pipeline for training
    W1, b1, W2 = get_init_params()
    m = X.shape[1]
    for i in range(iterations):
        pred_probs = np.zeros((2, m))
        for j in range(X.shape[1]-1):
            x1, x2 = np.reshape(X[:, j], (4,1)), np.reshape(X[:, j+1], (4,1))
            A1, A2, O1, O2 = forward(W1, b1, W2, x1, x2)
            dW1, db1, dW2 = backward(A1, A2, O2, W1, W2, x1, x2, Y[j+1], m)
            W1, b1, W2 = update_params(W1, b1, W2, dW1, db1, dW2, alpha)
            if j == 0:
                pred_probs[0, 0] = O1[0, 0]
                pred_probs[1, 0] = O1[1, 0]
            pred_probs[0, j+1] = O2[0, 0]
            pred_probs[1, j+1] = O2[1, 0]
        if (i+1) == iterations or i == 0:
            print("Iteration:", i+1)
            predictions = predict(pred_probs)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2

def predict_test(X, W1, b1, W2):
    # Pipeline for testing, similar to training but without backprop and update params
    pred_probs = np.zeros((2, X.shape[1]))
    for j in range(X.shape[1]-1):
        x1, x2 = np.reshape(X[:, j], (4,1)), np.reshape(X[:, j+1], (4,1))
        A1, A2, O1, O2 = forward(W1, b1, W2, x1, x2)
        if j == 0:
            pred_probs[0, 0] = O1[0, 0]
            pred_probs[1, 0] = O1[1, 0]
        pred_probs[0, j+1] = O2[0, 0]
        pred_probs[1, j+1] = O2[1, 0]
    predictions = predict(pred_probs)
    return predictions

def get_metrics(preds, Y):
    # Compute the precision, recall, and f-1 score
    tp, tn, fp, fn = 0, 0, 0, 0
    for pred, ans in zip(preds, Y):
        if pred == 0 and ans == 0:
            tn += 1
        if pred == 0 and ans == 1:
            fn += 1
        if pred == 1 and ans == 0:
            fp += 1
        if pred == 1 and ans == 1:
            tp += 1
    precision, recall = tp / (tp+fp), tp / (tp+fn)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

# Preprocess
data = spark.read.option("inferSchema", "true").option("header", "true").csv("/FileStore/tables/SMS_train.csv")
data = data.rdd.map(lambda x: (x['date'], x['precipitation'], x['temp_max'], x['temp_min'], x['wind'], x['weather'], get_label(x['weather']))).toDF(["date", 'precipitation', "temp_max", "temp_min", "wind", "weather", "label"])

train = data.limit(1200)
test = data.subtract(train).orderBy("date")

X = np.array(train.select("precipitation", "temp_max", "temp_min", "wind").collect())
X = X.T
Y = np.array(train.select("label").collect()).flatten()
Y = Y.T

X_test = np.array(test.select("precipitation", "temp_max", "temp_min", "wind").collect())
X_test = X_test.T
Y_test = np.array(test.select("label").collect()).flatten()
Y_test = Y_test.T

# Train and test
W1, b1, W2 = train(X, Y, 0.01, 50)
test_pred = predict_test(X_test, W1, b1, W2)
print(get_accuracy(test_pred, Y_test))

# Experiments
for lr in [0.001, 0.005, 0.01, 0.05]:
    for iter in [10, 50, 100, 500]:
        print("--- lr", lr, "iter", iter, "---")
        W1, b1, W2 = train(X, Y, lr, iter)
        test_pred = predict_test(X_test, W1, b1, W2)
        print("acc:", get_accuracy(test_pred, Y_test))
        print("p, r, f:", get_metrics(test_pred, Y_test))