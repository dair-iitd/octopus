from greedy import *
from fractional import *
from randomized import *
from randomized_robin import *

class Controller(object):

    def __init__(self,qualityPOMDPs,type=1,purpose=0,bundle=None):
        self.type = type
        if type == 1:
            if purpose == 0:
                self.algorithm = GreedySelect(1,[])
            elif purpose == 1:
                self.algorithm = GreedySelect(1,[],cache=1,bundle=bundle)
        elif type == 2:
            self.algorithm = FractionalSelect([])
        elif type == 3:
            self.algorithm = RandomSelect([])
        elif type == 4:
            self.algorithm = RandomRobinSelect([])

        self.cached = False
        if purpose == 1:
            self.cached = True
            self.cache = {}
            possibleV,alpha,beta,qualityPOMDP = bundle
            for v in possibleV:
                qualityPOMDP.qualityPOMDPBelief.setBelief((v + 1)/2.0,alpha,beta)
                self.cache[round((v + 1)/2.0,3)] = qualityPOMDP.findBestAction()

        self.resetController(qualityPOMDPs)

    '''
    Returns a list of k questions that are being assigned.
    '''
    def assignQuestion(self,k=1):
        if not self.questionsLeft:
            return []
        listOfQuestions = []
        for _ in xrange(k):
            question = self.algorithm.getQuestion()
            self.postedQuestions.add(question)
            self.questionsLeft.discard(question)
            listOfQuestions.append(question)
        return listOfQuestions


    def addAvailableQuestion(self,qualityPOMDP,cached=None):
        self.postedQuestions.discard(qualityPOMDP.question.question_id)
        if qualityPOMDP.findBestAction() > 0:
            self.questionsLeft.discard(qualityPOMDP.question.question_id)
            return False
        self.questionsLeft.add(qualityPOMDP.question.question_id)
        if self.type == 1:
            self.algorithm.addQuestion(qualityPOMDP,cached=cached)
        else:
            self.algorithm.addQuestion(qualityPOMDP)
        return True

    def addAvailableQuestions(self,qualityPOMDPs,cached=None):
        qualityPOMDPsLeft = []
        for qualityPOMDP in qualityPOMDPs:
            self.postedQuestions.discard(qualityPOMDP.question.question_id)
            if cached is None and self.cached == True:
                bestAction = self.cache[round(qualityPOMDP.qualityPOMDPBelief.v_max,3)]
            else:
                bestAction = qualityPOMDP.findBestAction()
            if bestAction > 0:
                self.questionsLeft.discard(qualityPOMDP.question.question_id)
            else:
                self.questionsLeft.add(qualityPOMDP.question.question_id)
                qualityPOMDPsLeft.append(qualityPOMDP)
                if self.type != 1:
                    self.algorithm.addQuestion(qualityPOMDP)

        if self.type == 1:
            self.algorithm.addQuestions(qualityPOMDPsLeft,cached=cached)


    def resetController(self,qualityPOMDPs):
        self.questionsLeft = set([qualityPOMDP.question.question_id for qualityPOMDP in qualityPOMDPs])
        self.postedQuestions = set()
        self.algorithm.reset()
        self.addAvailableQuestions(qualityPOMDPs)


    def recomputeController(self,qualityPOMDPs):
        self.algorithm.reset()
        qualityPOMDPsLeft = []
        for qualityPOMDP in qualityPOMDPs:
            if qualityPOMDP.question.question_id not in self.postedQuestions:
                qualityPOMDPsLeft.append(qualityPOMDP)
        self.addAvailableQuestions(qualityPOMDPsLeft)











