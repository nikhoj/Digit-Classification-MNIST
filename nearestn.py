import numpy as np
#import matplotlib.pyplot as plt

#train data
with open("train-images-idx3-ubyte.idx3-ubyte", 'rb') as train_images:
    trainX = bytearray(train_images.read())[16:]
    trainX = np.array(trainX).reshape((60000,784))
    
with open("train-labels.idx1-ubyte" , 'rb') as train_label:
    trainY = bytearray(train_label.read())[8:]
    trainY = np.array(trainY).reshape((60000,1))

mnist_train = trainX.copy()
mnist_train = np.append(mnist_train, trainY , axis = 1)

one = mnist_train[mnist_train[:,-1] == 1, :]
idx = np.random.randint(0, one.shape[0], 200)
sample1 = one[idx,:]
two = mnist_train[mnist_train[:,-1] == 2, :]
idx = np.random.randint(0, two.shape[0], 200)
sample2 = two[idx,:]
seven = mnist_train[mnist_train[:,-1] == 7, :]
idx = np.random.randint(0, seven.shape[0], 200)
sample7 = seven[idx,:]

train = np.append(sample1, np.append(sample2,sample7, axis = 0), axis = 0)
np.random.shuffle(train)

#split the trainset into train and validation
v = train[0:120]
x1 = train[120:]


#test k=1
count = 0
for i in range(0,120):
    vdata = v[:,0:-1]
    vdata = vdata[i,:]
    x1data = x1[:,0:-1]
    d = np.sqrt(((x1data - vdata)**2).sum(axis = 1))
    predict= x1[np.argsort(d)[0]][-1]
    count += 1 if predict == v[i,-1] else 0
acc = count / 120
print(acc)

#test k=3
count = 0
for i in range(0,120):
    vdata = v[:,0:-1]
    vdata = vdata[0]
    x1data = x1[:,0:-1]
    d = np.sqrt(((x1data - vdata)**2).sum(axis = 1))
    nearest = x1[np.argsort(d)[0:7]][:,-1]
    #predict = np.argsort(d)[-3:]
    bcount = np.bincount(nearest)
    predict = np.argmax(bcount)
    count += 1 if predict == v[i,-1] else 0
acc = count / 120
print(acc)




def bfclassifier(traindata, validation_vector, k = 1):
    label = traindata[:, 784]
    trainX = traindata[:,0:-1]
    vector = validation_vector[0:-1]
    d = np.sqrt(((trainX-vector)**2).sum(axis = 1))
    predict = label[np.argsort(d)[0]]
    return predict



