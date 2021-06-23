import numpy as np
import matplotlib.pyplot as plt

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

problem2()
















