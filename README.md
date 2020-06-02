## Classification in Python
The implemented classify functions are:
* Pegasos_V1,
* Pegasos_V2,
* Perceptron,
* SDCA,
* SGD.

All of the implemented algorithms can be used to classify data sets.  

### Requirement
```
python==3.7.0
numpy==1.18.1
```
### How to use

Open test.py you will find some examples

```
import numpy as np
import src.classifier as classifier

X = np.array([[3,1],[4,1],[4,2],[5,3],[5,4],[6,1],[6,3],[7,2],\
              [2,3],[2,4],[3,3],[3,4],[3,5],[4,4],[4,5],[7,4]])

Y = np.array([-1,-1,-1,-1,-1,-1,-1,-1,\
              +1,+1,+1,+1,+1,+1,+1,+1])

cs = classifier.CLASSIFIER(x = X,
                           y = Y,
                           T = 10000,
                           lr_rate=1e-3)
w = cs.Perceptron()
a = cs.plot(w)
```

### Experiments
We use matplotlib to draw the points X and their labels Y.  
The followings are the result from each classifier.  

#### SVM: Pegasos_V1 Classifier
![Pegasos_V1](./images/pegasos_v1.png)

#### SVM: Pegasos_V2 Classifier
![Pegasos_V2](./images/pegasos_v2.png)

#### Perceptron Classifier
![Perceptron](./images/perceptron.png)

#### SDCA Classifier
![SDCA](./images/sdca.png)

#### SGD Classifier
![SGD](./images/sgd.png)
