from numpy import random
'''
Class that characterizes the worker skill distribution. 
The mean is used as the 'average' worker who will do a task by the POMDP.
'''
class WorkerDistribution(object):

    def __init__(self,k,theta):
        self.k = k
        self.theta = theta
        self.mean = k*theta

    def generateWorker(self):
        return random.gamma(self.k,self.theta)

    def getMean(self):
        return self.mean
