import numpy as np
#import matplotlib.pyplot as plt
def problem1():
    with open("train-images-idx3-ubyte.idx3-ubyte", 'rb') as train_images:
        trainX = bytearray(train_images.read())[16:]
        trainX = np.array(trainX).reshape((60000,784))
        trainX[trainX > 0] = 1
    
    with open("train-labels.idx1-ubyte" , 'rb') as train_label:
        trainY = bytearray(train_label.read())[8:]
        trainY = np.array(trainY).reshape((60000,1))
        
    mnist_train = trainX.copy()
    mnist_train = np.append(mnist_train, trainY , axis = 1)
    
    #initialize the counter
    xcount = np.ones((784,10))
    ycount = np.ones((10))
    
    for data in mnist_train:
        label = data[784]
        ycount[label] += 1
        xcount[:,label] += data[0:-1].reshape((784))
        
    #finding py and px
    py = ycount/ycount.sum()
    px = xcount / ycount.reshape((1,10))
    
    #taking log
    logpx = np.log(px)
    logpxneg = np.log(1-px)
    logpy = np.log(py)
    
    #load test data
    with open('t10k-images-idx3-ubyte.idx3-ubyte', 'rb') as test_images:
        testX = bytearray(test_images.read())[16:]
        testX = np.array(testX).reshape((10000,784))
        testX[testX > 0] = 1
    
    with open('t10k-labels-idx1-ubyte.idx1-ubyte' , 'rb') as test_label:
        testY = bytearray(test_label.read())[8:]
        testY = np.array(testY).reshape((10000,1))
        
    mnist_test = testX.copy()
    mnist_test = np.append(mnist_test, testY , axis = 1)
    
    #posterior is proportional to likelihood X prior
    
    
    def bayespost(data):
        logpost = logpy.copy()
        logpost += (logpx * data + logpxneg * (1-data)).sum(0)
        logpost -= np.max(logpost)
        post = np.exp(logpost)
        #post /= np.sum(post)
        return np.argmax(post)
    
    tcount = 0
    for data in mnist_test:
        x = data[0:-1].reshape((784,1))
        y = int(data[-1])
        post = bayespost(x)
        tcount += (post == y)
    
    accuracy = round(((tcount/10000) * 100),2)
    print("Naive Bayes accuracy (with Dirichlet prior):" + str(accuracy))

problem1()
    
    
