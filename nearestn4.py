import numpy as np
import matplotlib.pyplot as plt

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
    
    

problem3()        
    

 






    