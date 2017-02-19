from AgentHuntReleaseOriginal.ModelLearning.utils import *
from AgentHuntReleaseOriginal.ModelLearning.genPOMDP import *
from AgentHuntReleaseOriginal.Data import *
from AgentHuntReleaseOriginal.Ballots import *
from quality_pomdp import *
from quality_pomdp_policy import *
from question import *
from worker_distribution import *
from difficulty_distribution import *
from system_parameters import *

from os import getcwd
from copy import deepcopy
import matplotlib.pyplot as plt
import cPickle as pickle

def main():#mt, numStates, numberOfProblems, nameOfTask, value, price,ZMDPPATH, URL, EMPATH, fastLearning, timeLearning,taskDuration, debug = False):
    #Do Stuff
    print "Do Stuff"
    test_expectedNumberOfBallotsToCompletion()
    #test_frontier_finding_approach()


def test_expectedNumberOfBallotsToCompletion():
    systemParameters = SystemParameters(DifficultyDistribution(2.0,2.0),WorkerDistribution(2.0,0.5))
    average_gamma = systemParameters.averageGamma
    ballots_to_completion_data = []
    repetitions = 200
    questions = []
    policy = QualityPOMDPPolicy(systemParameters,value=100,price=2)
    for i in xrange(repetitions):
        question = Question()
        question.question_difficulty = systemParameters.difficultyDistribution.generateDifficulty()
        question.question_answer = 0
        questions.append(question)
        qualityPOMDP = QualityPOMDP(question,policy)
        print "Repetition %d" % (i)
        ballots_to_completion_data.append([])
        ballot_count = 0
        submit = False
        while not submit:
            # Get action from POMDP, take a ballot, update belief, repeat
            m2 = qualityPOMDP.estimateBallotsToCompletionBySimulatingPOMDP(simulations=100,mode=2)
            # m2 = (0,0)
            ff = qualityPOMDP.estimateBallotsToCompletionUsingFrontierFinding(10)
            curr_belief = qualityPOMDP.qualityPOMDPBelief.belief
            best_action = qualityPOMDP.findBestAction()
            if best_action == 0:# and other_best_action == 0:
                ballot_count += 1
                print "Taking ballot %d." % (ballot_count)
                ballot = generateBallot(systemParameters.workerDistribution.generateWorker(),question.question_difficulty,question.question_answer)
                qualityPOMDP.qualityPOMDPBelief.updateBelief(ballot,average_gamma)
                ballots_to_completion_data[-1].append((m2,ff,curr_belief,ballot))
            else:
                submit = True

        print "Ballots Taken"
        print ballot_count
    with open(SystemParameters.path + "QualityPOMDP/tests/ExpectedNumberOfBallotsToCompletion/Paper_1.test",'w') as f:
        pickle.dump((systemParameters.workerDistribution,systemParameters.difficultyDistribution,100,2,ballots_to_completion_data),f)


def test_frontier_finding_approach():
    worker_distribution = WorkerDistribution(4.0,0.4)
    #worker_distribution = WorkerDistribution(32.0,0.05)

    question = Question(1)
    question.question_difficulty = 0.8
    question.question_answer = 0

    qualityPOMDP = QualityPOMDP(question,value=300,price=3)
    qualityPOMDP.learnPOMDP(worker_distribution.getMean())

    #otherqualityPOMDP = QualityPOMDP(question,value=300,price=4)
    #otherqualityPOMDP.learnPOMDP(worker_distribution.getMean())

    average_gamma = worker_distribution.getMean()
    ballots_to_completion_data = []
    repetitions = 1
    for i in xrange(repetitions):
        ballot_count = 0
        submit = False
        while not submit:
            # Get action from POMDP, take a ballot, update belief, repeat
            print "Expected Number of Ballots to Completion"
            print qualityPOMDP.estimateBallotsToCompletionUsingFrontierFinding(average_gamma,0)
            #print qualityPOMDP.belief

            #print otherqualityPOMDP.ballots_to_completion_frontier_finding(average_gamma,0)
            #print otherqualityPOMDP.belief

            best_action = qualityPOMDP.findBestAction()
            #other_best_action = otherqualityPOMDP.findBestAction()

            if best_action == 0:# and other_best_action == 0:
                ballot_count += 1
                print "Taking ballot %d." % (ballot_count)
                #ballot = generateBallot(average_gamma,self.question.question_difficulty,self.question.question_answer)
                ballot = generateBallot(worker_distribution.generateWorker(),question.question_difficulty,question.question_answer)
                #self.updateBeliefKeepingDifficultyFixed(ballot,average_gamma,difficulty)
                qualityPOMDP.updateBelief(ballot,average_gamma)
                #otherqualityPOMDP.updateBelief(ballot,average_gamma)
            else:
                submit = True

        print "Ballots Taken"
        print ballot_count

main()