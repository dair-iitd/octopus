import random

class RandomRobinSelect(object):

    def __init__(self,qualityPOMDPs):
        self.questions = []
        self.robins = set()
        for qualityPOMDP in qualityPOMDPs:
            self.addQuestion(qualityPOMDP)

    '''
    Add a question which is now available to be allocated to an incoming worker.
    '''
    def addQuestion(self,qualityPOMDP):
        if qualityPOMDP.qualityPOMDPBelief.v_max == 0.5:
            self.robins.add(qualityPOMDP.question.question_id)
        else:
            self.questions.append(qualityPOMDP.question.question_id)
            random.shuffle(self.questions)



    '''
    Select a question randomly only if round-robin is finished. Returns the questionID.
    '''
    def getQuestion(self,debug=False):
        if self.robins:
            return self.robins.pop()
        return self.questions.pop()


    def reset(self):
        self.questions = []
        self.robins = set()