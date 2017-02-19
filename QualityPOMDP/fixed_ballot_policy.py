import sys
sys.path.insert(0, '../')

from system_parameters import *

class FixedBallotPolicy(object):

    policyType = 2

    def __init__(self,systemParameters,numBallots):

        self.systemParameters = systemParameters
        self.numBallots = numBallots

