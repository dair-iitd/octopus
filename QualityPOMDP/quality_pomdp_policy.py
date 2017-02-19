# from AgentHuntReleaseOriginal.ModelLearning.utils import *
# from AgentHuntReleaseOriginal.ModelLearning.genPOMDP import *
# from AgentHuntReleaseOriginal.Data import *
# from AgentHuntReleaseOriginal.Ballots import *

import sys
sys.path.insert(0, '../')

from system_parameters import *
from itertools import product
from quality_pomdp_belief import getDifficulties


class QualityPOMDPPolicy(object):

    policyType = 0

    numStates = 23
    numDiffs = 11
    numActions = 3

    debug = False
    fastLearning = False
    timeLearning = 900

    if sys.platform == 'darwin':
        ZMDPPath = SystemParameters.path + 'QualityPOMDP/AgentHuntReleaseOriginal/ModelLearning/zmdp-1.1.7/bin/darwin/zmdp'
    else:
        ZMDPPath = SystemParameters.path + 'QualityPOMDP/AgentHuntReleaseOriginal/ModelLearning/zmdp-1.1.7/bin/linux3/zmdp'
    actions = range(0,3)
    difficulties = getDifficulties(0.1)

    def __init__(self,systemParameters,value=100,price=1):

        self.systemParameters = systemParameters

        self.value = value
        self.price = price
        self.policy = None

        self.learnPOMDP()

    def learnPOMDP(self):
        #POMDP is learned assuming that ballots will arrive using the current estimate of the average gamma value.
        #We can choose to update this value or we can simply assume that this will remain the mean of the
        #assumed distribution that workers follow.
        #print "Generating/Reading Policy"
        ###########################################################
        #Read the policy that we will begin with
        #You can choose a policy that's already been learned or learn
        #a new one
        ###########################################################
        os.chdir(SystemParameters.path + 'QualityPOMDP/')
        if os.path.exists('log/pomdp/%d_%4.2f_%4.2f.policy' % (self.value,self.price,self.systemParameters.averageGamma)):
            self.readPolicy('log/pomdp/%d_%4.2f_%4.2f.policy' % (self.value,self.price,self.systemParameters.averageGamma))
            return

        print "Learning Quality POMDP policy."
        self.genPOMDP('log/pomdp/%d_%4.2f_%4.2f.pomdp' % (self.value,self.price,self.systemParameters.averageGamma), [self.systemParameters.averageGamma])
        #Solve the POMDP
        zmdpDumpfile = open('log/pomdp/%d_%4.2f_%4.2f.zmdpDump' % (self.value,self.price,self.systemParameters.averageGamma), 'w')
	command = '%s solve %s -o %s -t %d' % (
                            QualityPOMDPPolicy.ZMDPPath,
                            SystemParameters.path + 'QualityPOMDP/log/pomdp/%d_%4.2f_%4.2f.pomdp' % (self.value,self.price,self.systemParameters.averageGamma),
                            SystemParameters.path + 'QualityPOMDP/log/pomdp/%d_%4.2f_%4.2f.policy' % (self.value,self.price,self.systemParameters.averageGamma),
                            QualityPOMDPPolicy.timeLearning)
        subprocess.call(command,
                        stdout=zmdpDumpfile,
                        shell=True,
                        executable="/bin/bash")
        zmdpDumpfile.close()
	print "Learned!"
        #Read the policy that we will begin with
	os.chdir(SystemParameters.path + 'QualityPOMDP/')
        self.readPolicy('log/pomdp/%d_%4.2f_%4.2f.policy' % (self.value,self.price,self.systemParameters.averageGamma))
        ################################################################
        # If you want a policy that we've already learned
        # pick one from ModelLearning/
        # W2R100 means 2 workflows, value of answer is 100
        #################################################################
        #policy = readPolicy("ModelLearning/Policies/W2R100.policy",
        #                    numStates)
        #print "POMDP Generated, POMDP Solved, Policy read"

    ############################
    # There are (numDiffs) * 2 states +  1 terminal state at the end.
    # We index as follows: Suppose Type A is the 0th difficulty,
    # type B is the 5th difficulty, and the answer is zero.
    # Then, the corresponding state number is
    # (0 * numDiffs * numDiffs) + (0 * numDiffs) + 5.
    #
    # Essentially, the first half of the states represents answer zero
    # The second half represents answer one
    # Each half is divided into numDiffs sections, representing
    # each possible difficulty for a typeA question.
    # Then each section is divided into numDiffs sections, representing
    # each possible difficulty for a typeB question.
    ###########################

    def genPOMDP(self, filename, gammas):

        #Add one absorbing state
        file = open(filename, 'w')
        file.write('discount: 0.9999\n')
        file.write('values: reward\n')
        file.write('states: %d\n' % QualityPOMDPPolicy.numStates)
        file.write('actions: %d\n' % QualityPOMDPPolicy.numActions)
        SUBMITZERO = 1
        SUBMITONE = 2
        file.write('observations: Zero One None\n')

        #Taking Action "Ask for ballot" always keeps you in the same state
        for i in range(0, QualityPOMDPPolicy.numStates):
            file.write('T: %d : %d : %d %f\n' % (0, i, i, 1.0))

        #Add transitions to absorbing state
        file.write('T: %d : * : %d %f\n' % (SUBMITZERO, QualityPOMDPPolicy.numStates-1, 1.0))
        file.write('T: %d : * : %d %f\n' % (SUBMITONE, QualityPOMDPPolicy.numStates-1, 1.0))

        #Add observations in absorbing state
        file.write('O: * : %d : None %f\n' % (QualityPOMDPPolicy.numStates-1, 1.0))

        for v in range(0, 2):
            for diffState in product(range(QualityPOMDPPolicy.numDiffs), repeat = 1):
                state = v * QualityPOMDPPolicy.numDiffs
                for k in range(0, 1):
                    state += (diffState[k] * (QualityPOMDPPolicy.numDiffs ** (1 - (k+1))))
                file.write('O: %d: %d : None %f\n' % (SUBMITZERO, state, 1.0))
                file.write('O: %d: %d : None %f\n' % (SUBMITONE, state, 1.0))
                if v == 0: #if the answer is 0
                    for k in range(0, 1):
                        file.write('O: %d : %d : Zero %f\n' %
                                   (k, state,
                                    calcAccuracy(gammas[k], QualityPOMDPPolicy.difficulties[diffState[k]])))
                        file.write('O: %d : %d : One %f\n' %
                                   (k, state, 1.0 - calcAccuracy(gammas[k],
                                                                 QualityPOMDPPolicy.difficulties[diffState[k]])))
                else: # if the answer is 1
                    for k in range(0, 1):
                        file.write('O: %d : %d : Zero %f\n' %
                                   (k, state,
                                    1.0 - calcAccuracy(gammas[k], QualityPOMDPPolicy.difficulties[diffState[k]])))
                        file.write('O: %d : %d : One %f\n' %
                                   (k, state, calcAccuracy(gammas[k],
                                                           QualityPOMDPPolicy.difficulties[diffState[k]])))

        file.write('R: * : * : * : * %f\n' % (-1 * self.price))

        for i in range(0, QualityPOMDPPolicy.numStates-1):
            if i < (QualityPOMDPPolicy.numStates-1) / 2:
                file.write('R: %d : %d : %d : * %f\n' % (SUBMITZERO, i, QualityPOMDPPolicy.numStates-1, 1))
                file.write('R: %d : %d : %d : * %f\n' % (SUBMITONE, i, QualityPOMDPPolicy.numStates-1, -1*self.value))
            else:
                file.write('R: %d : %d : %d : * %f\n' % (SUBMITONE, i, QualityPOMDPPolicy.numStates-1, 1))
                file.write('R: %d : %d : %d : * %f\n' % (SUBMITZERO, i, QualityPOMDPPolicy.numStates-1, -1*self.value))

        #Add rewards in absorbing state
        file.write('R: * : %d : %d : * %f\n' % (QualityPOMDPPolicy.numStates-1, QualityPOMDPPolicy.numStates-1, 0))
        file.close()


    def readPolicy(self,pathToPolicy):
        policy = {}
        lines = open(pathToPolicy, 'r').read().split("\n")

        numPlanes = 0
        action = 0
        alpha = [0 for k in range(0, QualityPOMDPPolicy.numStates)]
        insideEntries = False
        for i in range(0, len(lines)):
            line = lines[i]
            #First we ignore a bunch of lines at the beginning
            if (line.find('#') != -1 or line.find('{') != -1 or
                line.find('policyType') != -1 or line.find('}') != -1 or
                line.find('numPlanes') != -1 or
                ((line.find(']') != -1) and not insideEntries) or
                line.find('planes') != -1 or line == ''):
                continue
            if line.find('action') != -1:
                words = line.strip(', ').split(" => ")
                action = int(words[1])
                continue
            if line.find('numEntries') != -1:
                continue
            if line.find('entries') != -1:
                insideEntries = True
                continue
            if (line.find(']') != -1) and insideEntries: #We are done with one alpha vector
                if action not in policy:
                    policy[action] = []
                policy[action].append(alpha)
                action = 0
                alpha = ['*' for k in range(0, QualityPOMDPPolicy.numStates)]
                numPlanes += 1
                insideEntries = False
                continue
            #If we get here, we are reading state value pairs
            entry = line.split(",")
            state = int(entry[0])
            val = float(entry[1])
            alpha[state] = val
        #print "Policy Read"
        self.policy = policy
