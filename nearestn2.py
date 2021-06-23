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

one = mnist_train[mnist_train[:,-1] == 5, :]
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
v = train[480:]
x1 = train[0:480]


# split in 5 folds
#fold = validation , train
fold1 = [train[0:120], train[120:]]
fold2 = [train[120:240], np.append(train[0:120], train[240:], axis = 0)]
fold3 = [train[240:360], np.append(train[0:240], train[360:], axis = 0)]
fold4 = [train[360:480], np.append(train[0:360], train[480:], axis = 0)]
fold5 = [train[480:], train[0:480]]


#take predicted label for each fold
plabel = np.zeros((600, 5))



count = 0
for i in range (0,120):
    k = 3
    vector = fold1[0][i,0:784]
    label = fold1[0][i,784]
    x = fold1[1][:,0:784]
    xlabel = fold1[1][:,784]
    d = np.sqrt(np.square((x - vector)).sum(axis = 1))
    #d = np.linalg.norm(x1data[0,:] - vdata)
    nearest = np.argsort(d)[0:k]
    nearest = xlabel[nearest]
    #predict = np.argsort(d)[-3:]
    bcount = np.bincount(nearest)
    predict = np.argmax(bcount)
    
    #fin = predict == label
    #print(predict)
    #print(fin)
    count += 1 if predict == label else 0
acc = count / 120
print(acc)


'''
from scipy.spatial import distance
a = (1, 2, 3)
b = (4, 5, 6)
dst = distance.euclidean(a, b)
'''





