import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools as itr

def sample(distribution):
    return np.random.choice(distribution.size, 1, p=distribution)[0]
def softmax(q, T):
    x = np.exp(q / T)
    return x / np.sum(x)
# Convert q-value to *deterministic* policy using epsilon-greedy choice
def epsgreedy(q, eps=0.01):
    x = np.zeros(q.size)
    k = np.argmax(q) if np.random.sample() >= eps else np.random.randint(q.size)
    x[k] = 1
    return x
# Convert q-value to *stochastic* policy using epsilon-greedy choice, breaking ties
def epsgreedy2(q, eps=0.1):
    x = np.zeros(q.size)
    if np.random.sample() < eps:
        x[np.random.randint(q.size)] = 1
    else:
        x[q == np.max(q)] = 1
        x = x / np.sum(x)
    return x

pdA = np.array([[1, 4],
                [0, 3]])
pdB = np.array([[1, 0],
                [4, 3]])

class TwoDIQL():
    def __init__(self,A=pdA,B=pdB):
        self.A=A
        self.B=B
        pdA = np.array([[1, 4],
                        [0, 3]])
        pdB = np.array([[1, 0],
                        [4, 3]])

    def learn(self):
        # pdA = np.array([[1, -1],
        #                [-1, 1]])
        # pdB = np.array([[-1, 1],
        #                [1, -1]])

        # Constants
        discount = 0.0
        learn_rate = 0.001
        eps = 0.1
        N = 10**5


        seed = 4669201
        #np.random.seed(seed)

        payoffA = self.A
        payoffB = self.B

        print("Parameters:")
        print("  seed =", seed)
        print("  discount =", discount)
        print("  learn_rate =", learn_rate)
        print("  epsilon =", eps)
        print("  payoff matrix =", str(payoffA).replace("\n", "\n                  "))
        print("  Iterations, N =", N)

        # Initialize
        qA = np.array([2.8,2.801])
        qB = np.array([2.846,2.99])
        policyA = np.zeros(2)
        policyB = np.zeros(2)
        avg_payoffA = 0
        avg_payoffB = 0
        X = np.zeros(N)
        Y = np.zeros(N)
        P = np.zeros((N, 2, 2))
        Q = np.zeros((N, 2, 2))
        #fig = plt.figure()
        for k in range(N):
            if k % (N/10) == 0 :
                print("k=",k)
            #print("qA=",qA.size)
            policyA = epsgreedy(qA, eps)
            #print("pol",policyA)
            policyB = epsgreedy(qB, eps)
            a = sample(policyA)
            b = sample(policyB)
            rewardA = payoffA[a,b]
            rewardB = payoffB[a,b]
            qA[a] += learn_rate * (rewardA + discount * np.max(qA) - qA[a])
            qB[b] += learn_rate * (rewardB + discount * np.max(qB) - qB[b])
            avg_payoffA += (rewardA - avg_payoffA) / (k+1)
            avg_payoffB += (rewardB - avg_payoffB) / (k+1)
            X[k] = avg_payoffA
            Y[k] = avg_payoffB
            P[k, 0, :] = policyA
            P[k, 1, :] = policyB
            Q[k, 0, :] = qA
            Q[k, 1, :] = qB
        return learn_rate, eps,N,Q,P,X,Y


class DiscGIQLRandomPlot():

    def random_matrices(self,M=5,a=0,b=1):

        for i in range(0,M):
            print("This is for the {}th matrix".format(M))
            A = (b-a)*np.random.rand(2,2) + a
            B = A.T
            self.plot(A,B)
        plt.show()

    def plot(self,A,B):
        learn_rate,eps,N,Q,P,X,Y = TwoDIQL(A,B).learn()
        qA,qB = Q[:,0,:],Q[:,1,:]
        plt.plot(learn_rate*np.linspace(0,N,N),qA[:,0],'r:',lw=0.3)
        plt.plot(learn_rate*np.linspace(0,N,N),qA[:,1],'y--',lw=0.5)
        plt.plot(learn_rate*np.linspace(0,N,N),qB[:,0],'b:',lw=0.3)
        plt.plot(learn_rate*np.linspace(0,N,N),qB[:,1],'g--',lw=0.5)
        plt.title(r'GIQL being ran discretely for $\alpha = {},\epsilon$ = {}'.format(learn_rate,eps))

DiscGIQLRandomPlot().random_matrices(100,0,4)
