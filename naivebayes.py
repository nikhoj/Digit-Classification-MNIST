import numpy as np
import matplotlib.pyplot as plt


def problem2():
    with open("train-images-idx3-ubyte.idx3-ubyte", "rb") as train_image_f:
        train_image = bytearray(train_image_f.read())[16:]
        train_image = np.array(train_image).reshape((60000,784))
    with open("train-labels.idx1-ubyte", "rb") as train_lab_f:
        train_lab = bytearray(train_lab_f.read())[8:]
        train_lab = np.array(train_lab).reshape((60000,1))    
    train_lab5_idx = np.array(np.where(train_lab==5))[0]
    train_lab_5_idx = np.array(np.where(train_lab!=5))[0]
    mask = np.random.randint(0,2,size=len(train_lab5_idx), dtype=bool)
    train_img5_rep = train_image[train_lab5_idx]
    train_img5_rep = train_img5_rep[mask][:1000]
    train_img_5_rep = train_image[train_lab_5_idx]
    mask = np.random.randint(0,2,size=len(train_img_5_rep), dtype=bool)
    train_img_5_rep = train_img_5_rep[mask][:1000]
    labels = np.append(np.ones(1000), np.zeros(1000)).reshape((2000,1))
    data = np.append(train_img5_rep, train_img_5_rep, axis=0)
    data = np.append(data, labels, axis=1)
    np.random.shuffle(data)
    train = data[:1800]
    test = data[1800:]
    class_, counts = np.unique(train[:,784], return_counts = True)
    priors = np.vstack((class_, counts)).T
    priors[:,1] = priors[:,1]/priors[:,1].sum()
    mu0 = np.array((-9))
    mu1 = np.array((-9))
    pdf = lambda x,mu, v: np.log(v) + ((x-mu)**2/v)
    
    for i in range(28*28):
        mu0 = np.append(mu0, np.mean(train[train[:,784]==0][:,i]))
        mu1 = np.append(mu1, np.mean(train[train[:,784]==1][:,i]))
    mu0 = mu0[1:]
    mu1 = mu1[1:]
    v0 = np.var(train[train[:,784]==0])
    v1 = np.var(train[train[:,784]==1])
    def nvgaus(data, priors, mu0, mu1, v0, v1, tau):
        predict = []
        for image in data:
            for label in range(2):
                p = np.log(priors[label,1])
                v = v0 if label == 0 else v1
                for pixel in range(28*28):
                    mean = mu0[pixel] if label == 0 else mu1[pixel]
                    p += pdf(image[pixel], mean, v)
                p *=-0.5
                p -= (28*14) * np.log(2*np.pi)
                if label == 0:
                    likelihood_0 = p
                else:
                    likelihood_1 = p
            if (likelihood_1 - likelihood_0) >  tau:
                argmax_class = 1
            else:
                argmax_class = 0 
            predict.append(argmax_class)   
        predict = np.array(predict)
        return predict
    def get_t_fpr(data, tau, priors=priors, mu0=mu0, mu1=mu1, v0=v0, v1=v1):
        pred = nvgaus(data, priors, mu0, mu1, v0, v1, tau)
        data = data[:,784]
        tp = data[np.logical_and(data==1, data==pred)].shape[0]
        N_pos = data[np.logical_and(data==1, data==pred)].shape[0] + data[np.logical_and(data==1, data!=pred)].shape[0]
        fp = data[np.logical_and(data==0, data!=pred)].shape[0]
        N_neg = data[np.logical_and(data==0, data==pred)].shape[0] + data[np.logical_and(data==0, data!=pred)].shape[0]
        fpr = fp/N_neg
        tpr = tp/N_pos
        return tpr,fpr
    
    taus = np.linspace(0.1, 50, 50)
    x,y = [1], [1]
    for tau in taus:
        r = get_t_fpr(data = test,tau = tau)
        x.append(r[1])
        y.append(r[0])
    #x.append(0)
    #y.append(0)    
    x = np.array(x)
    y = np.array(y)
    plt.plot(x,y)
    
problem2()