'''
 *  Author : Bao Jiarong
 *  Contact: bao.salirong@gmail.com
 *  Created  On: 2020-06-01
 *  Modified On: 2020-06-01
 '''
from .algo import *
import numpy as np

class CLASSIFIER(Algo):
    #----------------------------------------------------------
    # Constructor
    #----------------------------------------------------------
    def __init__(self,x,y,T,lr_rate=1e-3):
        Algo.__init__(self,x,y,T,lr_rate)
        self.x   = x
        self.y   = y
        self.T   = T
        self.lr_rate  = lr_rate


    def PegasosV1(self):
        w = np.zeros(self.x.shape[-1])
        L = len(self.x)

        for t in range(1,self.T):
            i = np.random.randint(L)
            x = self.x[i]
            y = self.y[i]
            zeta = 1.0 / (t * self.lr_rate)
            score = w @ x
            if y * score < 1:
                w = (1 - zeta * self.lr_rate) * w + (zeta * y) * x
            else:
                w = (1 - zeta * self.lr_rate) * w
        return w

    def PegasosV2(self,f):
        n = len(self.x)
        w = np.zeros(self.x.shape[-1])
        t = 0

        for t in range(1,self.T):
            t = t + 1
            i = int(random.uniform(0,n-1))
            η = (t*self.lr_rate)**-1
            loss = f(self.x[i],self.y[i],w)
            if loss < 1 :
                w = (1 - η*self.lr_rate)*w + η*self.y[i]*self.x[i]
            else:
                w = (1 - η*self.lr_rate)*w
        return w

    def Perceptron(self):
        n = len(self.x)
        w = r(self.x.shape[-1])
        for epoch in range(1,self.T):
            i = int(random.uniform(0,n-1))
            y1 = np.tanh(w@self.x[i])
            df = (self.y[i]-y1)*self.x[i]
            w = w + self.lr_rate*df
        return w

    def SGD(self,f):
        n = len(self.x)
        w = np.zeros(self.x.shape[-1])
        for epoch in range(1,self.T):
            i = int(random.uniform(0,n-1))
            loss = f(self.x[i],self.y[i],w)
            if loss < 1 :
                w = w + self.lr_rate*self.y[i]*self.x[i]
        return w

    def SDCA(self,f):
        n = len(self.x)
        w = np.zeros(self.x.shape[-1])
        alpha = np.zeros(n)
        for epoch in range(1,self.T):
            I = random_set(n)
            for j in range(n-1):
                i = I[j]
                d_alpha = f(self.x[i],self.y[i],w,alpha[i],self.lr_rate,n)
                alpha[i] = alpha[i] + d_alpha
                w = w + ((self.lr_rate*n)**-1)*d_alpha*self.x[i]
        return w
