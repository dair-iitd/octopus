import heapq
import random

class RandomSelect(object):

    def __init__(self,qualityPOMDPs):
        self.questions = []
        for qualityPOMDP in qualityPOMDPs:
            self.addQuestion(qualityPOMDP)

    '''
    Add a question which is now available to be allocated to an incoming worker.
    '''
    def addQuestion(self,qualityPOMDP):
        self.questions.append(qualityPOMDP.question.question_id)
        random.shuffle(self.questions)


    '''
    Select a question randomly. Returns the questionID.
    '''
    def getQuestion(self,debug=False):
        return self.questions.pop()

    def reset(self):
        self.questions = []