import sys
sys.path.insert(0, '../')

import numpy as np
from system_parameters import *
import subprocess, os

class WorkerSkillEstimator(object):

    EMPath = SystemParameters.path + 'QualityPOMDP/EM/em'

    def __init__(self,systemParameters):
        self.threshold = 4
        self.ballots = {}
        self.workersToIDs = {}
        self.IDsToWorkers = {}
        self.workersToGammas = {}
        self.workersToNumberOfResponses = {}
        self.numWorkers = 0
        self.numBallots = 0
        self.numQuestions = 0
        self.systemParameters = systemParameters


    def addBallot(self,ballot,workerID,questionID):
        if questionID not in self.ballots:
            self.ballots[questionID] = []
            self.numQuestions += 1
        if workerID not in self.workersToIDs:
            self.workersToIDs[workerID] = self.numWorkers
            self.IDsToWorkers[self.numWorkers] = workerID
            self.workersToGammas[self.numWorkers] = self.systemParameters.averageGamma
            self.workersToNumberOfResponses[self.numWorkers] = 0
            self.numWorkers += 1
        self.numBallots += 1
        self.ballots[questionID].append((ballot,workerID))
        self.workersToNumberOfResponses[self.workersToIDs[workerID]] += 1


    def writeToEMFormat(self,idx):
        with open('log/em/ballots.eminput%d' % (idx), 'w') as f:
            f.write('%d %d %d %d %f\n' % (self.numBallots, self.numWorkers, self.numQuestions, 1, 0.5))
            for i,questionID in enumerate(self.ballots.keys()):
                for (ballot, workerID) in self.ballots[questionID]:
                    f.write('%d %d %d %d\n' % (i, self.workersToIDs[workerID], 0, ballot))

    def runEM(self,idx):
        outputfile = open('log/em/emresults%d' % (idx), 'w')
        subprocess.call('%s%d log/em/ballots.eminput%d' % (self.EMPath,idx,idx), stdout=outputfile, shell=True)
        outputfile.close()

        #Now we want to format the output to be nice
        gammaoutputs = open('log/em/gammas.emresults%d' % (idx), 'w')
        diffoutputs = open('log/em/diffs.emresults%d' % (idx), 'w')
        posterioroutputs = open('log/em/posteriors.emresults%d' % (idx),'w')

        inputfile = open('log/em/emresults%d' % (idx), 'r').read().split("\n")
        inputfile = inputfile[0:len(inputfile) - 1]
        problemCount = 0
        workerCount = 0
        restart = False
        for i in range(0, len(inputfile)):
            nextLine = inputfile[i]
            if 'nan' in nextLine:
                restart = True
                break
            if 'Iteration' in nextLine:
                continue
            elif 'Beta' in nextLine: #These are problem difficulties
                d = float(nextLine.split("=")[1].strip())
                diffoutputs.write('%f\n' % d)
                problemCount += 1 #These are posteriors
            elif 'P' in nextLine:
                p = float(nextLine.split("=")[2].strip())
                posterioroutputs.write('%f\n' % p)
            else: #These are worker gammas.
                g = float(nextLine)
                gammaoutputs.write('%f\n' % g)
                workerCount += 1

        gammaoutputs.close()
        diffoutputs.close()
        posterioroutputs.close()

        if restart:
            self.runEM(idx)

    def relearnGammas(self):
        os.chdir(SystemParameters.path + 'QualityPOMDP')
        idx = 0
        while True:
            try:
                os.mkdir('locks/lockEM%d' % (idx))
                break
            except OSError:
                idx = (idx + 1) % (100)
                pass
        if self.numWorkers > 0:
            self.writeToEMFormat(idx)
            self.runEM(idx)
            self.readGammaResults(idx)
        os.chdir(SystemParameters.path + 'QualityPOMDP')
        os.rmdir('locks/lockEM%d' % (idx))
        #self.systemParameters.averageGamma = self.calcAverageGammas()

    def readGammaResults(self,idx):
        infile = open('log/em/gammas.emresults%d' %(idx), 'r')
        infile = infile.read().split("\n")
        infile = infile[0:len(infile) - 1]
        for i in range(0, len(infile)):
            if self.workersToNumberOfResponses[i] >= self.threshold:
                self.workersToGammas[i] = float(infile[i])

    def getWorkerGamma(self, workerId):
        if workerId in self.workersToIDs:
            return self.workersToGammas[self.workersToIDs[workerId]]
        else:
            gamma = self.calcAverageGammas()
            return gamma

    def calcAverageGammas(self):
        gammas = [value for key,value in self.workersToGammas.items()]
        if len(gammas) == 0:
            avg = self.systemParameters.workerDistribution.getMean()
        else:
            avg = np.average(gammas)
        return avg

