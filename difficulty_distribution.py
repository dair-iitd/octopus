from numpy import random
'''
Class that characterizes the difficulty distribution from which
tasks are drawn. E.g. - used as a prior for the POMDP.
'''
class DifficultyDistribution(object):

    def __init__(self,alpha,beta):

        self.alpha = alpha
        self.beta = beta

    def generateDifficulty(self):
        return random.beta(self.alpha,self.beta)

    def getMean(self):
        return self.alpha/float(self.alpha+self.beta)
