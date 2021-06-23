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
#v = train[480:]
#x1 = train[0:480]


# split in 5 folds
#fold = validation , train
fold1 = [train[0:120], train[120:]]
fold2 = [train[120:240], np.append(train[0:120], train[240:], axis = 0)]
fold3 = [train[240:360], np.append(train[0:240], train[360:], axis = 0)]
fold4 = [train[360:480], np.append(train[0:360], train[480:], axis = 0)]
fold5 = [train[480:], train[0:480]]

v1 , x1 = fold1
v2 , x2 = fold2
v3 , x3 = fold3
v4 , x4 = fold4
v5 , x5 = fold5


#take predicted label for each fold
plabel = np.zeros((120, 4))

test = v1[4,0:784]
trn = x1[:,0:784]
trnlabel = x1[:,784]


def NNclassifier(testdata, traindata, trainlabel, k):
    i = 0
    predict = np.zeros((testdata.shape[0], 1))
    for image in testdata:
    
        test = np.tile(image, (480,1))
        dist = (trn - test)**2
        dist = np.sum(dist, axis =1)
        dist = np.sqrt(dist)
        
        nearest = trnlabel[np.argsort(dist)[0:k]]
        bcount = np.bincount(nearest)
        predict[i] = np.argmax(bcount)
        i += 1
    return predict

NNclassifier(test, trn, trnlabel, k = 3)
    




n = trn.shape[0] 
diffMat = np.tile(test, (n, 1)) - trn
sqDiffMat = diffMat ** 2
sqDistances = sqDiffMat.sum(axis=1)
distances = sqDistances ** 0.5  
sortedDistIndicies = distances.argsort()  





