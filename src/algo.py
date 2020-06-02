'''
 *  Author : Bao Jiarong
 *  Contact: bao.salirong@gmail.com
 *  Created  On: 2020-06-01
 *  Modified On: 2020-06-01
 '''
import random
import numpy as np
import matplotlib.pyplot as plt

def r(n):
    a = []
    for i in range(n):
        a.append(random.uniform(0,1))
    return a

def random_set(n):
    a = []
    for i in range(n):
        a.append(int(random.uniform(0,n-1)))
    return a

class Algo:
    #----------------------------------------------------------
    # Constructor
    #----------------------------------------------------------
    def __init__(self,x,y,T,lr_rate=1e-3):
        self.x   = x
        self.y   = y
        self.T   = T
        self.lr_rate  = lr_rate

    def plot(self,w):
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        n = self.x.shape[0]
        for i in range(n):
            if self.y[i]==-1:
                x1.append(self.x[i][0])
                y1.append(self.x[i][1])
            else:
                x2.append(self.x[i][0])
                y2.append(self.x[i][1])

        fig= plt.figure()
        ax = fig.add_axes([0.1,0.1,0.85,0.85])
        ax.grid(color='b', ls = '-.', lw = 0.25)
        ax.set_xlabel("x axis")
        ax.set_ylabel("y axis")
        ax.set_xlim(-8,8)
        ax.set_ylim(-8,8)
        xx = np.arange(-8,9)
        yy = np.arange(-8,9)
        ax.plot(xx,0*xx)
        ax.plot(0*yy,yy)
        ax.scatter(x1, y1, color='b')
        ax.scatter(x2, y2 , color='r')
        yy = (-w[0]*xx)/w[1]
        ax.plot(xx,yy)
        plt.show()
