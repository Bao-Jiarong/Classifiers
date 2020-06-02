'''
 *  Author : Bao Jiarong
 *  Contact: bao.salirong@gmail.com
 *  Created  On: 2020-06-01
 *  Modified On: 2020-06-01
 '''
import numpy as np
import src.classifier as classifier

def hinge_loss(xi,yi,w):
    w = np.array(w)
    w = np.transpose(w)
    t = (xi@w)*yi
    return t

def lipschitz_hinge_loss(xi,yi,w,ai,lr_rate,n):
    t = (1-xi@w*yi)/(((xi**2).sum())/(lr_rate*n))+ai*yi
    t1 = min(1,t)
    loss = (yi*max(0,t1))-ai
    return loss

def lipschitz_deviation_loss(xi,yi,w,ai,lr_rate,n):
    t = (yi-xi@w)/(((xi**2).sum())/(lr_rate*n))+ai
    t1 = min(1,t)
    loss = max(-1,t1)-ai
    return loss
# -------------------------------------------------------------
X = np.array([[3,1],[4,1],[4,2],[5,3],[5,4],[6,1],[6,3],[7,2],\
              [2,3],[2,4],[3,3],[3,4],[3,5],[4,4],[4,5],[7,4]])

Y = np.array([-1,-1,-1,-1,-1,-1,-1,-1,\
              +1,+1,+1,+1,+1,+1,+1,+1])

cs = classifier.CLASSIFIER(x = X,           # data set
                           y = Y,           # data set
                           T = 10000,       # maximum number of iterations
                           lr_rate=1e-3)    # learning rate
# --------------------------------------------------------------
print("PegasosV1")
w = cs.PegasosV1()
a = cs.plot(w)
# -------------------------------------------------------------
print("PegasosV2")
w = cs.PegasosV2(f = hinge_loss)
a = cs.plot(w)
# -------------------------------------------------------------
print("Perceptron")
w = cs.Perceptron()
a = cs.plot(w)
# -------------------------------------------------------------
print("SDCA")
w = cs.SDCA(lipschitz_hinge_loss)     # other:lipschitz_deviation_loss
a = cs.plot(w)
# -------------------------------------------------------------
print("SGD")
w = cs.SGD(f = hinge_loss)
a = cs.plot(w)
