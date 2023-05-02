from pyspark.sql.functions import col
import numpy as np


###########################################################
# AWS public data file setting
###########################################################
spark.sparkContext._jsc.hadoopConfiguration().set('fs.s3a.access.key','AKIAXMYK3Q64GI24T45G')
spark.sparkContext._jsc.hadoopConfiguration().set('fs.s3a.secret.key','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')
spark.sparkContext._jsc.hadoopConfiguration().set('fs.s3a.endpoint','s3.amazonaws.com')
input_path = 's3://kylbucket/seattle-weather.csv'


###########################################################
# separate the features into labels
# Bad weather = 1
# Good weather = 0
###########################################################
def get_label(x):
    if x == "rain" or x == "drizzle" or x == "snow":
        return 1
    else:
        return 0

###########################################################
# Provide random values for initial parameters
###########################################################
def get_init_params():
    # Set initial weights to be -0.1 ~ 0.1
    W1 = np.random.uniform(low=-0.1, high=0.1, size=(2, 4))
    bias = np.random.uniform(low=-0.1, high=0.1, size=(2, 1))
    W2 = np.random.uniform(low=-0.1, high=0.1, size=(2, 2))
    return W1, bias, W2

###########################################################
# transform input matrix using softmax function 
###########################################################
def softmax(X):
    # Shift values to avoid overflow
    f = np.exp(X-np.max(X))  
    return f / f.sum(axis=0)


###########################################################
# covert probabilities Y ==> 0 or 1
###########################################################
def one_hot_encoding(result):
    # Convert label to one-hot encoding, ex: 1 -> [0 1]
    new_result = np.zeros((2, 1))
    new_result[result, 0] = 1
    return new_result


###########################################################
# update W1, bias, W2 for new set using alpha parameter
###########################################################
def update_parameters(W1,  W2, bias, delta_W1,  delta_W2, delta_bias, alpha):
    return W1 - alpha * delta_W1, bias - alpha * delta_bias, W2 - alpha * delta_W2 

###########################################################
# calcaulte forward propagation values
###########################################################
def forward(W1, W2, bias, x1, x2):
    #calculate previous day weight values
    A1 = W1.dot(x1) + bias
    O1 = softmax(A1) # transformation
    
    # calculate current day values with previous day values
    A2 = W1.dot(x2) +  W2.dot(A1) + bias 
    O2 = softmax(A2) # transformation 
    return A1, A2, O1, O2

###########################################################
# calcaulte backward propagation values
###########################################################
def backward(A1, A2, O2, W1, W2, x1, x2, y2, m):
    # Backward propogation in RNN
    y = one_hot_encoding(y2)    #update values to 0 or 1
    delta_A2 = O2 - y           
    delta_W2 = delta_A2.dot(A1.T) / m
    delta_W1 = (delta_A2.dot(x2.T) + W2.T.dot(delta_A2).dot(x1.T)) / m
    delta_bias = (delta_A2 + W2.T.dot(delta_A2)) / m
    return delta_W1, delta_bias, delta_W2

###########################################################
# get predicted labels
###########################################################
def predict(probs):
    # Convert probabilities to predicted labels
    Pred_Y = np.argmax(probs, 0)
    return Pred_Y

###########################################################
# calculate accuracy of resulting labels
###########################################################
def accuracy(preds, Y):
    return np.sum(preds == Y) / Y.size

###########################################################
# train the model using gradient descent 
###########################################################
def train(X, Y, alpha, iterations):
    # The pipeline for training
    W1, bias, W2 = get_init_params()
    m = X.shape[1]
    for i in range(iterations):
        pred_probs = np.zeros((2, m))
        for j in range(X.shape[1]-1):
            # X1 = previous day & X2 = present day 
            x1, x2 = np.reshape(X[:, j], (4,1)), np.reshape(X[:, j+1], (4,1)) 
            
            # first run to get forward propagation value
            A1, A2, O1, O2 = forward(W1,  W2, bias, x1, x2)
            
            # backward propagation
            delta_W1, delta_bias, delta_W2 = backward(A1, A2, O2, W1, W2, x1, x2, Y[j+1], m)
            
            # update new parameters
            W1, bias, W2 = update_parameters(W1,  W2, bias, delta_W1, delta_W2, delta_bias, alpha)
            
            #create output matrix
            if j == 0:
                pred_probs[0, 0] = O1[0, 0] #initial matrix for good weather
                pred_probs[1, 0] = O1[1, 0] #initial matrix for bad weather
            pred_probs[0, j+1] = O2[0, 0]   #matrix for good weather
            pred_probs[1, j+1] = O2[1, 0]   #matrix for bad weather
        if (i+1) == iterations or i == 0:  #print training accuracy
            print("Iteration:", i+1)
            iter_predictions = predict(pred_probs)
            print(accuracy(iter_predictions, Y))
    return W1, bias, W2


###########################################################
# calculate testing model using parameters from training model
###########################################################
def predict_test(X, W1, bias, W2):
    # Pipeline for testing, similar to training but without backprop and update params
    pred_probs = np.zeros((2, X.shape[1]))
    for j in range(X.shape[1]-1):
        x1, x2 = np.reshape(X[:, j], (4,1)), np.reshape(X[:, j+1], (4,1))
        A1, A2, O1, O2 = forward(W1, W2, bias, x1, x2) #calculate forward propagation 
        if j == 0:
            pred_probs[0, 0] = O1[0, 0] #initial matrix for good weather
            pred_probs[1, 0] = O1[1, 0] #initial matrix for bad weather
        pred_probs[0, j+1] = O2[0, 0] #matrix for good weather
        pred_probs[1, j+1] = O2[1, 0]  #matrix for bad weather
    predictions = predict(pred_probs)  #print training accuracy
    return predictions

###########################################################
# calculate precision, recall, F-1 score
###########################################################
def metrics(predicted, Y):
    # Compute the precision, recall, and f-1 score
    true_p, true_n, false_p, false_n = 0, 0, 0, 0
    for predicted, actual in zip(predicted, Y):
        if predicted == 0 and actual == 0: #true negative
            true_n = true_n + 1
        if predicted == 0 and actual == 1: #false negative
            false_n = false_n + 1
        if predicted == 1 and actual == 0: #false positive
            false_p = false_p + 1
        if predicted == 1 and actual == 1: #true positive
            true_p = true_p + 1
    precision = true_p/(true_p + false_p) #calculate precision
    recall =  true_p/(true_p + false_n) #calculate recall 
    f1 = 2 * precision * recall / (precision + recall) #calculate F1-score
    return precision, recall, f1

###########################################################
# preprocess data
###########################################################
data = spark.read.option("inferSchema", "true").option("header", "true").csv(input_path)
data = data.rdd.map(lambda x: (x['date'], x['precipitation'], x['temp_max'], x['temp_min'], x['wind'], x['weather'], get_label(x['weather']))).toDF(["date", 'precipitation', "temp_max", "temp_min", "wind", "weather", "label"])

###########################################################
# split data into training and testing data sets and transpose matrix
###########################################################
training_set = data.limit(1200)
testing_set = data.subtract(training_set).orderBy("date")

X = np.array(training_set.select("precipitation", "temp_max", "temp_min", "wind").collect())
X = X.T
Y = np.array(training_set.select("label").collect()).flatten()
Y = Y.T

X_test = np.array(testing_set.select("precipitation", "temp_max", "temp_min", "wind").collect())
X_test = X_test.T
Y_test = np.array(testing_set.select("label").collect()).flatten()
Y_test = Y_test.T

###########################################################
# Train and test data sets
###########################################################
W1, bias, W2 = train(X, Y, 0.01, 50) #create training model
test_pred = predict_test(X_test, W1, bias, W2) #test prediction model
print(accuracy(test_pred, Y_test))

###########################################################
# Experiments
# iterations: 10, 50, 100, 500 
# learning rates: 0.001, 0.005, 0.01, 0.05
###########################################################
for lr in [0.001, 0.005, 0.01, 0.05]:
    for iter in [10, 50, 100, 500]:
        print("--- lr", lr, "iter", iter, "---")
        W1, bias, W2 = train(X, Y, lr, iter) #train model
        test_pred = predict_test(X_test, W1, bias, W2) #run model
        print("acc:", accuracy(test_pred, Y_test)) #get accuracy value
        print("p, r, f:", metrics(test_pred, Y_test)) #get precision, recall, and f-1 score
