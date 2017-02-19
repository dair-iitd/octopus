from system_parameters import *


class Question(object):

    def __init__(self,price=-1,question_id=None,question_status=0,question_answer=-1,question_difficulty=None,is_gold_question=False,simulation=True,question_data=None,hit_id=None):
        self.question_id = SystemParameters.addQuestion() if question_id is None else question_id
        self.price = price
        self.question_status = question_status
        self.question_answer = question_answer
        self.question_difficulty = question_difficulty
        self.is_gold_question = is_gold_question
        self.simulation = simulation

        self.question_data = question_data

        self.hit_id = hit_id
        self.workers_used = []


