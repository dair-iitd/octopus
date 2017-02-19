import sys
sys.path.insert(0, '../')

from system_parameters import *

class FixedPolicy(object):

    policyType = 1

    def __init__(self,systemParameters,threshold):

        self.systemParameters = systemParameters
        self.threshold = threshold

