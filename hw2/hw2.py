#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

from sklearn.model_selection import train_test_split,StratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score


# # K-Means Clustering

# In[2]:


from sklearn.datasets import load_breast_cancer


# In[3]:


X,y = load_breast_cancer(return_X_y=True, as_frame=True)


# In[4]:


#No missing values
X.isnull().sum()


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X.values,y,test_size=0.3,random_state=42)


# In[6]:


def euclidian_distance(a,b):
    return math.sqrt(sum([(a[i]-b[i])**2 for i in range(len(a))]))


# In[7]:


def classify(X,centroids):
    c1,c2 = centroids
    return np.array([ 'A' if euclidian_distance(c1,X[i])<euclidian_distance(c2,X[i]) else 'B' for i in range(len(X))])


# In[8]:


def KMeans(X,k=2):
    c1,c2 = X[np.random.randint(0,len(X))], X[np.random.randint(0,len(X))]
    classes = classify(X,(c1,c2))
    while True:
        c1 = (np.sum(X[classes=='A'],axis=0) / len(X[classes=='A']))
        c2 = (np.sum(X[classes=='B'],axis=0) / len(X[classes=='B']))
        temp = classify(X,(c1,c2))
        if np.array_equal(temp,classes):
            break
        classes = temp
    return (classes, (c1,c2))


# In[9]:


def kmeans_predict(X,centroids):
    c1,c2 = centroids
    classes = classify(X,(c1,c2))
    return classes


# In[10]:


pred, centroids = KMeans(X_train)


# In[11]:


cluster_with_max = pd.value_counts(pred).keys()[0]


# In[12]:


label_with_max_values = pd.value_counts(y_train).keys()[0]


# In[13]:


cluster_with_max,label_with_max_values


# Because clustering cannot assign a class in the y_train, we are reassigning the most predicted cluster with the most predicted label using y_train.

# In[14]:


for i in range(10):
    test_preds = kmeans_predict(X_test,centroids)
    test_pred = np.where(test_preds == cluster_with_max, label_with_max_values,pd.value_counts(y_train).keys()[1])
    print(accuracy_score(y_test, test_pred)) 


# Running k-means 10 times with different centroids still results in the same accuracy because eventually all converge to the same centroids.

# In[15]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200)
rf.fit(X_train,y_train)
accuracy_score(y_test,rf.predict(X_test))


# Yes, using a supervised algorithm results in a better accuracy.

# # Gradient Descent

# In[16]:


def gradient_descent(gradient, fn, iterations=10, alpha=0.5, theta=np.array([0,0])):
    t = 1
    while t<=iterations:
        theta = theta-alpha*(gradient(theta[0],theta[1]))
        print('function value: {} at iteration {}'.format(fn(theta[0],theta[1]),t))
        t+=1
    return theta


# In[17]:


theta_1 = theta_2 = np.array([0,0])
alpha = 0.5
f1 = lambda x,y: (x-2)**2 + (y-3)**2
f2 = lambda x,y: (1-(y-3))**2 + 20*((x+3)-(y-3)**2)**2


grad_f1 = lambda x,y: np.array([2*(x-2),2*(y-3)])

grad_f2 = lambda x,y: np.array([40*(x+3-(y-3)**2), -2*(4-y)-(80*(-((y-3)**2)+x+3)*(y-3))])


# In[18]:


theta_1 = gradient_descent(grad_f1,f1)
print('Final value of theta where f1 is the lowest:{}'.format(theta_1))


# In[19]:


theta_2 = gradient_descent(grad_f2,f2,100)
print('Final value of theta where f2 is the lowest:{}'.format(theta_2))


# Because the learning rate is high and the function is a polynomial with degree 4 the gradient increases exponentially, which will not result in any solution. Using the second order approximation of gradient descent might result in better values.

# In[20]:


hessian_f2 = lambda x, y: np.linalg.inv(np.array([[40, -80*(y-3)],[-80*(y-3), 240*y*y-1440*y-80*x+1992]]))


# In[21]:


def gradient_descent_hessian(gradient, fn, iterations=10, alpha=0.5, theta=np.array([0,0])):
    t = 1
    while t<=iterations:
        theta = theta-alpha* np.dot(hessian_f2(theta[0],theta[1]),(gradient(theta[0],theta[1])))
        print('function value: {} at iteration {}'.format(fn(theta[0],theta[1]),t))
        t+=1
    return theta


# In[22]:


theta_2 = gradient_descent_hessian(grad_f2,f2,100)
print('Final value of theta where f2 is the lowest:{}'.format(theta_2))


# # Naive Bayes Classifier - 3rd Question

# In[23]:


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle
import math


# In[24]:


boys_names = pd.read_csv('3/boy_names.csv')
girls_names = pd.read_csv('3/girl_names-2.csv')
test_names = pd.read_csv('3/test_names.csv')


# In[25]:


b = boys_names.x
g = girls_names.x
test = pd.DataFrame(np.array(test_names.x), columns=['Names'])


# In[26]:


df = pd.DataFrame(np.append(b,g),columns=['Names'])
df['Gender'] = np.array([-1]*1000+[1]*1000)


# In[27]:


# Referred the following link to generate features
# https://github.com/clintval/gender-predictor/blob/master/gender_predictor/__init__.py
consonants = "bcdfghjklmnpqrstvwxyz"
def generate_features(X):
    X['Names'] = X['Names'].str.lower()
    X['Start']  = X['Names'].str[0]
    X['Last'] = X['Names'].str[-1]
    X['Starts_With_Vowel'] = [X['Start'][i] in 'AEIOU'.lower() for i in range(len(X))]
    X['Starts_With_Consonant'] = [X['Start'][i] in consonants for i in range(len(X))]
    X['Ends_With_Vowel'] = [X['Last'][i] in 'AEIOU'.lower() for i in range(len(X))]
    X['Ends_With_Consonant'] = [X['Last'][i] in consonants for i in range(len(X))]
    X['First_two'] = X['Names'].str[:2]
    X_trans = X.drop('Names',axis=1)
    cols = list(filter(lambda x:x!='Gender',list(X_trans.columns)))
    for col in cols:
        X_trans[col] = LabelEncoder().fit_transform(X[col])
    return X_trans


# In[28]:


X = generate_features(df)


# In[29]:


X.head(5)


# In[30]:


get_mean = lambda feature,gender,X: np.mean(X[X.Gender==gender][feature])


# In[31]:


get_variance = lambda feature,gender,X: np.var(X[X.Gender==gender][feature])


# In[32]:


def get_means_and_variance(X):
    means_male = {}
    means_female = {}
    var_male = {}
    var_female = {}
    for col in X.columns[1:]:
        means_male[col] = get_mean(col,-1,X)
        means_female[col] = get_mean(col,1,X)
        var_male[col] = get_variance(col,-1,X)
        var_female[col] = get_variance(col,1,X)
    return {'means_male':means_male,'means_female':means_female,'var_male':var_male,'var_female':var_female}


# In[33]:


X, y = shuffle(X,X.Gender)


# In[34]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[35]:


means_var = get_means_and_variance(X_train)


# In[36]:


X_train = X_train.drop('Gender',axis=1)
X_test = X_test.drop('Gender',axis=1)


# Assuming Gaussian Distribution and calculating the likelihoods.
# 
# P0 is the likelihood for class -1 i.e P(X | Y=-1) (-1 is Male).
# 
# P1 is the likelihood for class 1 i.e P(X | Y=1) (1 is Female).

# In[37]:


def get_likelihoods_and_preds(X):
    P0 = []
    P1 = []
    preds = []
    for x in X.values:
        p1 = 1
        p0 = 1
        for i in range(len(np.array(X.columns))):
                k = 1/(math.sqrt(2*math.pi*means_var['var_male'][X.columns[i]]))
                exp = -((x[i]-means_var['means_male'][X.columns[i]])**2)/(2*means_var['var_male'][X.columns[i]])
                p0 = p0*k*np.exp(exp) #Multiplying the likelihoods of each feature.

                k = 1/(math.sqrt(2*math.pi*means_var['var_female'][X.columns[i]]))
                exp = -((x[i]-means_var['means_female'][X.columns[i]])**2)/(2*means_var['var_female'][X.columns[i]])
                p1 = p1*k*np.exp(exp)

        P1.append(p1)
        P0.append(p0)
        preds.append(1 if p1>p0 else -1) #Assigning a class based on likelihoods. 
    return {'P1':P1,'P0':P0, 'preds':preds}


# In[38]:


X_trans = X.drop('Gender',axis=1)
d = get_likelihoods_and_preds(X_trans)
d['P1'] #P(X | Y=1)
d['P0']#P(X | Y=-1)


# In[39]:


accuracy_score(y_test,get_likelihoods_and_preds(X_test)['preds'])


# P(Y=1) and P(Y=-1) are the probabilities of either being a boy or girl. And they both are the same i.e 1/2.

# Log Probabilities. log(P(Y=1 | X) / P(Y=-1 | X)).
# 
# We know that,
# P(Y=1 | X) is proportional to P(X1 | Y).P(X2 | Y)...P(Xd | Y).P(Y), we can ignore P(Y) because its constant and same.
# 
# So,
# P(Y=1 | X) = P1 i.e Likelihood for class 1.
# 
# P(Y=-1 | X) = P0 i.e Likelihood for class -1

# In[40]:


P1 = np.array(d['P1'])
P0 = np.array(d['P0'])
log_prob = np.log(P1/P0)


# In[41]:


log_prob


# In[42]:


test_features = generate_features(test)


# In[43]:


test_features.head(5)


# In[44]:


preds = pd.DataFrame(get_likelihoods_and_preds(test_features)['preds'], columns=['Gender'])
preds.to_csv('test_preds.csv',index=False)


# # Naive Bayes Classifier - 4th Question

# Bayes assumption,
# P(X|Y = y) = P(X[1] | Y=y).P(X[2] | Y=y).....P(X[d] | Y=y).
# 
# 'd' is the number of features

# In[45]:


# Referred the following link 
# https://www.probabilitycourse.com/chapter1/1_4_4_conditional_independence.php


# Lets consider 3 events as follows,
# 
# A = {1,2,4}
# B = {2,5}
# C = {2,3}
# 
# P(A) = 1/3
# P(B) = 1/3
# Now, according to bayes assumption,
# P(A^B | C) = P(A|C) * P(B|C)
# 
# P(A^B) = {2}
# So, P(A^B | C} = 1/2
# 
# P(A|C) = 1/3, P(B|C) = 1/2
# So, P(A|C)* P(B|C) = 1/6
# 
# P(A^B|C) > P(A|C) * P(B|C)
# 
# Now, lets assume C = {1,5}
# 
# P(A|C) = 1/3
# P(B|C) = 1/2
# P(A^B|C) = 0
# 
# P(A^B|C) < P(A|C) * P(B|C)

# If the model is a full model, P(X | Y=c) can be written as follows,
# P(X | Y=c) = P(X1^X2^X3^....Xn) / P(Y=c).
# 
# Here P(X1^X2^X3...Xd) can be written as,
# P(Xn | X1,X2,...Xn-1)P(Xn-1 | X1,X2,X3...Xn-2)...P(X2|X1)P(X1). '^' is intersection.
# 
# If we assume that X is boolean vector here, then X can have 2^n possible values, where n is the the number of features.
# So, the number of parameters will approximately be 2^n.
# 
# If the training size is too small, Naive Bayes would perform better because it does not assume any dependence between the features and it can easily compute the probabilities, and with a full bayesian model it would be very difficult to train the model because it is hard to derive a correlation between features with small amounts of data. 

# In[ ]:




