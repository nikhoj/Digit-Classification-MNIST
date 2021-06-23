"""
CS5783: Machine Learning
Assignment 2
Submitted By: Md Mahabub Uz Zaman
A20099364
"""

import numpy as np
import matplotlib.pyplot as plt

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



def problem2():
    #train data
    with open("train-images-idx3-ubyte.idx3-ubyte", 'rb') as train_images:
        trainX = bytearray(train_images.read())[16:]
        trainX = np.array(trainX).reshape((60000,784))
    
    with open("train-labels.idx1-ubyte" , 'rb') as train_label:
        trainY = bytearray(train_label.read())[8:]
        trainY = np.array(trainY).reshape((60000,1))
        trainY[trainY != 5] = 0
        trainY[trainY == 5] = 1
        
    mnist_train = trainX.copy()
    mnist_train = np.append(mnist_train, trainY , axis = 1)
    
    #seperate based on label 5
    five = mnist_train[mnist_train[:,-1]==1]
    others = mnist_train[mnist_train[:,-1]==0]
    
    #selecting 1000 randomly
    idx = np.random.randint(five.shape[0], size = 1000)
    five = five[idx,:]
    idx = np.random.randint(five.shape[0], size = 1000)
    others = others[idx,:]
    
    #combining two together
    data = np.append(five, others, axis = 0)
    
    #randomly pick trainset and test set
    idx = np.random.randint(data.shape[0], size = int(.9 * data.shape[0]))
    traindata = data[idx,:]
    trainX = traindata[:,0:-1]
    trainY = traindata[:,-1]
    idx = np.random.randint(data.shape[0], size = int(.1 * data.shape[0]))
    testdata = data[idx,:]
    testX = testdata[:,0:-1]
    testY = testdata[:,-1]
    
    #calculating mu and variance
    N = 28 * 28
    mu = np.zeros((784,2))
    mu[:,0] = trainX[trainY == 0].sum(axis = 0) / N
    mu[:,1] = trainX[trainY == 1].sum(axis = 0) / N
    v = np.zeros((2))
    v[0] = np.var(trainX[trainY == 0])
    v[1] = np.var(trainX[trainY == 1])
    
    #py calculation
    
    ycount = np.ones((2))
    
    for data in traindata:
        label = data[784]
        ycount[label] += 1
    
    py = ycount/ ycount.sum()
    
    
    
    
    
    def pdf(data, mu, var):
      D = data.shape[0]
      prob = (np.log(var) + ((data-mu)**2/var)).sum()
      prob *= -0.5
      prob -= (D/2)* np.log(2*np.pi)
      return prob
    
    def predict(prior, data, mu, var, tau):
      L0 = np.log(prior[0])
      L0 += pdf(data, mu[:,0], v[0])
      L1 = np.log(prior[1])
      L1 += pdf(data, mu[:,1], v[1])
      prediction = (L1 - L0) > tau
      return prediction
      
    accuracy = []
    tau = np.linspace(.1,100,50)
    x = [1]
    y = [1]
    for tval in tau:
        count = 0
        FP = 0
        FN = 0
        TP = 0
        TN = 0
    
        for data in testdata:
            label = data[784]
            prediction = 1 if predict(py, data[0:784], mu, v, tau = tval) else 0
            count += 1 if prediction == label else 0
            TP += 1 if prediction == True and label == 1 else 0
            TN += 1 if prediction == False and label == 0 else 0
            FP += 1 if prediction == True and label == 0 else 0
            FN += 1 if prediction == False and label == 1 else 0
        accuracy.append(count / testdata.shape[0])
        TPR = round(TP / (TP + FN),4)
        y.append(TPR)
        FPR = round(FP / (FP + TN),4)
        x.append(FPR)
     
        
    plt.plot(x,y, color ='g')
    plt.ylabel('True Positive Rate (TPR)')
    plt.xlabel('False Positive Rate (FPR)')
    plt.title('ROC Curve')
    plt.show()
    
    #print(accuracy)


def problem3():
    #train data
    with open("train-images-idx3-ubyte.idx3-ubyte", 'rb') as train_images:
        trainX = bytearray(train_images.read())[16:]
        trainX = np.array(trainX).reshape((60000,784))
    
    #train Label    
    with open("train-labels.idx1-ubyte" , 'rb') as train_label:
        trainY = bytearray(train_label.read())[8:]
        trainY = np.array(trainY).reshape((60000,1))
    
    #combine the data with label
    mnist_data = np.append(trainX, trainY, axis = 1)
    
    #split based on 1, 2, 7
    rawdata1 = mnist_data[mnist_data[:,784] == 1]
    rawdata2 = mnist_data[mnist_data[:,784] == 2]
    rawdata7 = mnist_data[mnist_data[:,784] == 7]
    
    
    idx1 = np.random.randint(0, int(rawdata1.shape[0] * .8), 200)
    idx2 = np.random.randint(0, int(rawdata2.shape[0] * .8), 200)
    idx7 = np.random.randint(0, int(rawdata7.shape[0] * .8), 200)
    
    data1 = rawdata1[idx1,:]
    data2 = rawdata2[idx2,:]
    data7 = rawdata7[idx7,:]
    
    traindata = np.vstack((data1,data2,data7))
    np.random.shuffle(traindata)
    
    #split into five fold
    fold1 = [traindata[0:120], traindata[120:]]
    fold2 = [traindata[120:240], np.append(traindata[0:120], traindata[240:], axis = 0)]
    fold3 = [traindata[240:360], np.append(traindata[0:240], traindata[360:], axis = 0)]
    fold4 = [traindata[360:480], np.append(traindata[0:360], traindata[480:], axis = 0)]
    fold5 = [traindata[480:], traindata[0:480]]
    
    fold = [fold1, fold2, fold3, fold4, fold5]
    
    #test set split
    
    
    idx1 = np.random.randint(int(rawdata1.shape[0] * .8) , rawdata1.shape[0], 50)
    idx2 = np.random.randint(int(rawdata2.shape[0] * .8) ,rawdata2.shape[0], 50)
    idx7 = np.random.randint(int(rawdata7.shape[0] * .8) ,rawdata7.shape[0], 50)
    
    testdata1 = rawdata1[idx1,:]
    testdata2 = rawdata2[idx2,:]
    testdata7 = rawdata7[idx7,:]
    
    testdata = np.vstack((testdata1,testdata2,testdata7))
    np.random.shuffle(testdata)
    #Classifier build
    
    def dist(a, b):
        return np.array([np.sqrt(np.sum((a - bi) ** 2.)) for bi in b])
    
    def kNNclassifier(testvector, trainset, trainlabel, k):
    
        #calculate distance
        d = dist(testvector, trainset )
        
        #sort
        nearest = trainlabel[np.argsort(d)[0:k]]
        predict = np.bincount(nearest)
        predict = np.argmax(predict)
        
        return predict
    
    
    def accuracy(testset, trainset, k):
        #testimage = testset[:,0:784]
        #testlabel = testset[:,784]
        trainimage = trainset[:,0:784]
        trainlabel = trainset[:,784]
        n = testset.shape[0]
        count = 0
        
        for image in testset:
            imagedata = image[0:784]
            imagelabel = image[784]
            p = kNNclassifier(imagedata, trainimage, trainlabel, k)
            count += 1 if p == imagelabel else 0
            
        acc = count / n   
        return acc
    
    
    def find_best_k(k_range, fold):
        #fold_accuracy = np.zeros((5,5))
        avglist = np.zeros((5,1))
        fold1, fold2, fold3, fold4, fold5 = fold
        i = 0
        #Validation , trainset = fold
        v1 , x1 = fold1
        v2 , x2 = fold2
        v3 , x3 = fold3
        v4 , x4 = fold4
        v5 , x5 = fold5
        
    
        for k in k_range:    
        
            acc1 = accuracy(v1,x1,k)
            acc2 = accuracy(v2,x2,k)
            acc3 = accuracy(v3,x3,k)
            acc4 = accuracy(v4,x4,k)
            acc5 = accuracy(v5,x5,k)
            
            avg = (acc1+ acc2+ acc3+ acc4+ acc5) / 5
            avglist[i] = avg
            i += 1
        valid_acc = float(avglist[np.argmax(avglist)])
        bestk = k_range[np.argmax(avglist)]
        
        
        
        print("Best K = " + str(bestk))
        print("Validation accuracy = " + str(valid_acc) )
        
        
        return bestk
    kval = np.array([1,3,5,7,9])
    best_k = find_best_k(kval, fold)
    test_acc = accuracy(testdata,traindata,best_k )
    print("test accuracy = " + str(test_acc))
    
    #plot start from here
    
    def plotgen(testdata, traindata, status = True):
        
        for image in testdata:
            imagedata = image[0:784]
            imagelabel = image[784]
            trainimage = traindata[:,0:784]
            trainlabel = traindata[:,784]
            if imagelabel == kNNclassifier(imagedata, trainimage, trainlabel, 1) and status == True:
                
                return imagedata
                break
            
            elif imagelabel != kNNclassifier(imagedata, trainimage, trainlabel, 1) and status == False:
                
                return imagedata
                break
            
            
    
    R1 = plotgen(testdata1, traindata) 
    W1 = plotgen(rawdata1, traindata, status = False)
    R2 = plotgen(testdata2, traindata) 
    W2 = plotgen(testdata2, traindata, status = False)
    R7 = plotgen(testdata7, traindata) 
    W7 = plotgen(testdata7, traindata, status = False)
    
    plt.subplot(231)
    plt.title("Correct 1")
    plt.imshow(np.reshape(R1, (28,28)))
    plt.subplot(232)
    plt.title("Correct 2")
    plt.imshow(np.reshape(R2, (28,28)))
    plt.subplot(233)
    plt.title("Correct 7")
    plt.imshow(np.reshape(R7, (28,28)))
    plt.subplot(235)
    plt.title("Wrong 2")
    plt.imshow(np.reshape(W2, (28,28)))
    plt.subplot(236)
    plt.title("Wrong 7")
    plt.imshow(np.reshape(W7, (28,28)))
    plt.subplot(234)
    plt.title("Wrong 1")
    plt.imshow(np.reshape(W1, (28,28)))
    
    
    plt.subplots_adjust(hspace = .5)
    plt.show()
   
#problem1()
#problem2()
problem3()

    


