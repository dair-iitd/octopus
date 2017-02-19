# from AgentHuntReleaseOriginal.ModelLearning.utils import *
# from AgentHuntReleaseOriginal.ModelLearning.genPOMDP import *
# from AgentHuntReleaseOriginal.Data import *
# from AgentHuntReleaseOriginal.Ballots import *

import sys
sys.path.insert(0, '../')

from scipy import stats

def getDifficulties(diffInterval):
    difficulties = []
    numDiffs = int(1.0 / diffInterval + 1)
    for i in range(0, numDiffs):
        difficulties.append(round(diffInterval * i, 1))
    return difficulties

def normalize(array):

    sum = 0.0
    for i in range(0, len(array)):
        sum += array[i]
    for i in range(0, len(array)):
        array[i] = array[i] / sum
    return array

def calcAccuracy(gamma, d):
    return (1.0/2) * (1.0 + (1.0-d) ** gamma)

class QualityPOMDPBelief(object):

    numStates = 23
    numDiffs = 11

    difficulties = getDifficulties(0.1)

    def __init__(self,v=0.5,alpha=1.0,beta=1.0):

        self.belief = None
        self.v_max = None
        self.prediction = None
        self.ballots_taken = 0


        self.alpha = alpha
        self.beta = beta

        rv = stats.beta(self.alpha,self.beta)
        self.imposeDistribution = []
        for j in xrange(QualityPOMDPBelief.numDiffs):
            val = (rv.cdf((j + 0.5)/(QualityPOMDPBelief.numDiffs - 1)) - rv.cdf((j - 0.5)/(QualityPOMDPBelief.numDiffs - 1)))
            self.imposeDistribution.append(val)

        self.setBelief(v,self.alpha,self.beta)

    def calculateAccuracy(self,gamma):
        accuracy = 1
        for j in range(0, QualityPOMDPBelief.numDiffs):
            accuracy += (1 - QualityPOMDPBelief.difficulties[j])**gamma * (self.belief[j] + self.belief[QualityPOMDPBelief.numDiffs + j])
        return 0.5*accuracy


    def resetBelief(self):
        self.ballots_taken = 0
        self.setBelief(0.5,self.alpha,self.beta)

    def setBelief(self,v,a,b):
        self.belief = [0 for _ in range(QualityPOMDPBelief.numStates)]
        if (a == self.alpha and b == self.beta):
            for i in [0,1]:
                for j in xrange(QualityPOMDPBelief.numDiffs):
                    self.belief[i*QualityPOMDPBelief.numDiffs + j] = ((1-i) * v + i * (1-v)) * self.imposeDistribution[j]
        else:
            rv = stats.beta(a,b)
            for i in [0,1]:
                for j in xrange(QualityPOMDPBelief.numDiffs):
                    self.belief[i*QualityPOMDPBelief.numDiffs + j] = (1-i) * v * (rv.cdf((j + 0.5)/(QualityPOMDPBelief.numDiffs - 1)) - rv.cdf((j - 0.5)/(QualityPOMDPBelief.numDiffs - 1))) + \
                                                                     i * (1-v) * (rv.cdf((j + 0.5)/(QualityPOMDPBelief.numDiffs - 1)) - rv.cdf((j - 0.5)/(QualityPOMDPBelief.numDiffs - 1)))
                # if i == 0:
                #     self.belief[i*QualityPOMDPBelief.numDiffs + j] = (1-i) * v * (rv.cdf((j + 0.5)/(QualityPOMDPBelief.numDiffs - 1)) - rv.cdf((j - 0.5)/(QualityPOMDPBelief.numDiffs - 1)))
                # else:
                #     self.belief[i*QualityPOMDPBelief.numDiffs + j] = i * (1-v) * (rv.cdf((j + 0.5)/(QualityPOMDPBelief.numDiffs - 1)) - rv.cdf((j - 0.5)/(QualityPOMDPBelief.numDiffs - 1)))
        self.belief = normalize(self.belief)
        self.prediction,self.v_max = self.getAnswerInformation()



    def updateBelief(self, observation, gamma):
        #action must be 0 to enter this function
        newBeliefs = []
        for i in range(0, 2):
            for j in range(0, QualityPOMDPBelief.numDiffs):
                diffA = QualityPOMDPBelief.difficulties[j]
                state = (i * QualityPOMDPBelief.numDiffs) + j
                if observation == i:
                        newBeliefs.append(calcAccuracy(gamma, diffA) *
                                          self.belief[state])
                else:
                        newBeliefs.append((1-calcAccuracy(gamma, diffA)) *
                                          self.belief[state])

        newBeliefs.append(0.0)
        normalize(newBeliefs)
        self.belief = newBeliefs
        self.prediction,self.v_max = self.getAnswerInformation()

    def updateBeliefKeepingDifficultyFixed(self, observation, gamma, difficulty):
        #action must be 0 to enter this function
        newBeliefs = []
        beliefOverValues = [0,0]
        updatedBeliefOverValues = [0,0]
        for j in range(0, QualityPOMDPBelief.numDiffs):
            state = j
            beliefOverValues[0] += self.belief[state]

        beliefOverValues[1] = 1 - beliefOverValues[0]

        accuracy = calcAccuracy(gamma,difficulty)

        for i in range(0,2):
            if i == observation:
                updatedBeliefOverValues[i] = beliefOverValues[i] * accuracy / (beliefOverValues[i] * accuracy + beliefOverValues[(i + 1)%2] * (1 - accuracy))
            else:
                updatedBeliefOverValues[i] = beliefOverValues[i] * (1 - accuracy) / (beliefOverValues[i] * (1 - accuracy) + beliefOverValues[(i + 1)%2] * accuracy)

            for j in range(0,QualityPOMDPBelief.numDiffs):
                state = (i * QualityPOMDPBelief.numDiffs) + j
                newBeliefs.append(self.belief[state] * updatedBeliefOverValues[i] / beliefOverValues[i])

        newBeliefs.append(0.0)
        self.belief = newBeliefs
        self.prediction,self.v_max = self.getAnswerInformation()

    def getMostLikelyDifficulty(self):
        bestState = -1
        bestProb = 0
        for i in range(0, 2):
            for j in range(0, QualityPOMDPBelief.numDiffs):
                diffA = QualityPOMDPBelief.difficulties[j]
                state = (i * QualityPOMDPBelief.numDiffs) + j
                if self.belief[state] > bestProb:
                    bestState = diffA
                    bestProb = self.belief[state]
        return bestState

    def getAverageDifficulty(self):
        expected_difficulty = 0.0
        for j in range(0,QualityPOMDPBelief.numDiffs):
            s0 = j
            s1 = QualityPOMDPBelief.numDiffs + j
            expected_difficulty += QualityPOMDPBelief.difficulties[j]*(self.belief[s0] + self.belief[s1])

        return expected_difficulty

    def getMostLikelyAnswer(self):
        prob_0 = 0.0
        for j in range(0,QualityPOMDPBelief.numDiffs):
            prob_0 += self.belief[j]

        prob_1 = 1 - prob_0

        if prob_0 >= prob_1:
            return 0
        return 1

    def getBeliefInAnswer(self):
        prob_0 = 0.0
        for j in range(0,QualityPOMDPBelief.numDiffs):
            prob_0 += self.belief[j]

        prob_1 = 1 - prob_0

        return prob_0,prob_1

    def getAnswerInformation(self):
        prob_0 = 0.0
        for j in range(0,QualityPOMDPBelief.numDiffs):
            prob_0 += self.belief[j]

        prob_1 = 1 - prob_0

        if prob_0 >= prob_1:
            return 0, prob_0

        return 1, prob_1

