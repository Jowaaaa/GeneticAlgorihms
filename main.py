from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn_genetic.plots import plot_fitness_evolution
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import random
import time


data = datasets.load_diabetes

dataset = pd.read_csv('diabetes.csv')
#label encoder om M & B naar 1 & 0 te veranderen
LE = LabelEncoder()
dataset.iloc[:,1]=LE.fit_transform(dataset.iloc[:,1].values)

#dependent X & independent Y
X = dataset.iloc[:,0:7].values
Y = dataset.iloc[:,8].values

X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#setup Neural Network
maxNodes = 5 #5
hiddenLayers = 2 #2
networkIn = X_train.shape[1]
networkHidden1 = maxNodes
networkHidden2 = maxNodes
networkOut = 1


def MLPClass():
    #clf = MLPClassifier(random_state=1, max_iter=500).fit(X_train, Y_train)
    clf = MLPRegressor(random_state=1, max_iter=500).fit(X_train, Y_train)
    #clf.predict_proba(X_test[:1])

    y_pred = clf.predict(X_test) #[:5, :]
    print('Mean Squared Error MPLClassifier', mean_squared_error(Y_test,y_pred))

MLPClass()

def GenAlgo(networkIn, networkHidden1, networkHidden2, networkOut):
    #build Neural Network
    model = Sequential()
    model.add(Dense(networkHidden1, input_dim=networkIn, activation='relu'))
    model.add(Dense(networkHidden2, activation='relu'))
    model.add(Dense(networkOut, activation='linear'))

    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    
    return model 

GenAlgo(networkIn, networkHidden1, networkHidden2, networkOut)

def runEpi (X_train, Y_train, X_test, Y_test, networkIn, networkOut, policy):
    networkHidden1,networkHidden2 = policy
    model = GenAlgo(networkIn, networkHidden1, networkHidden2, networkOut)
    model.fit(X_train, Y_train, epochs=100, verbose=0, validation_split=0.05)
    _, accuracy = model.evaluate(X_test, Y_test)
    return (accuracy)

# policy = [5,2]
# acc1 = runepi(X_train, Y_train, X_test, Y_test, networkIn, networkOut, policy)

# policy = [2,5]
# acc2 = runepi(X_train, Y_train, X_test, Y_test, networkIn, networkOut, policy)

# policy = [5,5]
# acc3 = runepi(X_train, Y_train, X_test, Y_test, networkIn, networkOut, policy)

# print('accuracy:',  acc1, 'accuracy 2:', acc2, 'accuracy 3:', acc3)


def evaluatePolicy (X_train, Y_train, X_test, Y_test, networkIn, networkOut, policy, episodes=10):
    reward = 0.0
    for _ in range (episodes):
        reward += 1/runEpi(X_train, Y_train, X_test, Y_test, networkIn, networkOut, policy)
    return reward/episodes
    
def randomPolicy(maxNodes, hiddenLayers):
    return np.random.choice(maxNodes+1, size=((hiddenLayers)))


def crossover(policy1, policy2, hiddenLayers):
    newPolicy = policy1.copy()
    for i in range(hiddenLayers):
        rand = np.random.uniform()
        if rand > 0.5:
            newPolicy[i] = policy2[i]
    return newPolicy

def mutation(policy, hiddenLayers, maxNodes, p=0.05):
    newPolicy = policy.copy()
    for i in range(hiddenLayers):
        rand = np.random.uniform()
        if rand < p:
            newPolicy[i] = np.random.choice(maxNodes+1)
    return newPolicy


if __name__ == '__main__':
    random.seed(1234)
    np.random.seed(1234)

    #policy search
    numberPolicy = 10
    numberSteps = 5
    start = time.time()
    policy_pop = [randomPolicy(maxNodes,hiddenLayers) for _ in range(numberPolicy)]
    for idx in range (numberSteps):
        policyScores = [evaluatePolicy(X_train,Y_train,X_test,Y_test, networkIn, networkOut, p) for p in policy_pop]
        print('Generation %d : MaxScore=%0.2f & AvgScore=%0.3f' %(idx+1, max(policyScores), sum(policyScores)/len(policyScores)))
        policyRank = list(reversed(np.argsort(policyScores)))
        eliteSet = [policy_pop[x] for x in policyRank[:5]]
        selectProbs = np.array(policyScores) / np.sum(policyScores)
        if np.sum(policyScores)==0:
            pp = 1/np.array(policyScores).size
            selectProbs = pp*np.ones(np.array(policyScores).size)
        childSet = [crossover(
            policy_pop[np.random.choice(range(numberPolicy), p=selectProbs)],
            policy_pop[np.random.choice(range(numberPolicy), p=selectProbs)], hiddenLayers)
            for _ in range(numberPolicy - 5)]
        mutatedList = [mutation(p,hiddenLayers, maxNodes) for p in childSet]
        policy_pop = eliteSet
        policy_pop += mutatedList
    policyScore = [evaluatePolicy(X_train,Y_train, X_test, Y_test, networkIn, networkOut, p) for p in policy_pop]
    bestPolicy = policy_pop[np.argmax(policyScore)]

    end = time.time()
    print('Best policy score=%0.2f Time taken (sec)=%4.4f Average Score=%0.3f' %(np.max(policyScore), (end-start), sum(policyScore)/len(policyScore)))
    print('Best Policy is:', bestPolicy)

