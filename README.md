No commercial use, but you're welcome to the code or data for research purposes or personal use. Please continue to retain this README.

Please cite if you use the data/code.

@article{goel2017octopus,
  title={Octopus: A Framework for Cost-Quality-Time Optimization in Crowdsourcing},
  author={Karan Goel, Shreya Rajpal and Mausam},
  journal={arXiv preprint arXiv:1702.03488},
  year={2017}
}

Directory structure

- __init__.py

#These are simple helper classes.
- difficulty_distribution.py
- question.py
- worker_distribution.py

#This is the class that makes note of all parameters that we set when testing or running the system.
#Important! Open and set the path variable here.
- system_parameters.py

#This contains the TaskSelector code (called the controller in this code).
- Controller
	#The controller interface is in this file. Controllers are referred to by numbers
	#1: Greedy, 3: Random, 4:Random-Robin in the testbed.
	- controller.py
	#These are the 3 controller implementations used for Octopus.
	- greedy.py
	- randomized_robin.py
	- randomized.py

#Offline data collection code and data is in this folder.
- DataCollection
	# 0.01_repeat, 0.02_repeat, 0.03_repeat, 0.04, 0.05, 0.06 are the data that was collected
	# at the 6 price points that was used in the paper. Some data collection was repeated
	# due to weekend effects. Final data was all collected on weekdays starting at around 
	# 9-10am PT.
	- 0.01
	- 0.01_repeat
		- Results_1 #These Results_i files are what contain the actual results.
	- 0.02
	- 0.02_repeat
		- Results_2
	- 0.03
	- 0.03_repeat
		- Results_3
	- 0.04
		- Results_4
	- 0.05
		- Results_5
	- 0.06
		- Results_6

	#These 2 files contain code for posting HITs containing tweets to MTurk and fetching responses.
	- functions.py
	- main.py
	#This is the dataset we used taken from UT-Austin's SQUARE
	- square_twitter_data.csv


#We use this package to learn MDPs using ValueIteration.
- pymdptoolbox

- QualityPOMDP
	#This is a nicely abstracted form of the QualityPOMDP (QualityManagers). The code is very easy
	#to both read and use.
	- quality_pomdp.py
	- quality_pomdp_belief.py
	- quality_pomdp_policy.py
	#This is used to run the EM algorithm of Whitehill. Once again it's nice code.
	- worker_skill_estimation.py
	#This contains the EM algorithm of Whitehill. The key thing to note is that if you're running
	#this on a unix machine, you should carefully move em0 to a new folder (call it macintosh), and
	#move all the files in the unix subdirectory to its parent (i.e. the EM folder itself). The
	#reason there are these multiple em scripts (em0, em1 and so on) are to parallelize multiple 
	#simulations of Octopus without being bottlenecked by access to 1 em file.
	#If you need to make again, just use the Makefile. Make sure you make lots of em files again,
	#as done in the unix folder.
	- EM
		#Compiled on Macintosh
		- em0
		#Compiled on Ubuntu
		- unix
	#No need to touch this, used automatically to manage resources.
	- locks
	#Contains em input-output (don't bother about this), and POMDP policies (no need to worry about that
	#either).
	- log

#This is where the CostSetter code lies. 
- TMDP
	#This is the class that sets up everything for the CostSetter. It's commented reasonably
	#and intuitively set up so it shouldn't be hard to follow. Some things have been set to certain
	#values for heuristic/computational reasons, which shouldn't really be bothered about.
	- cost_mdp.py
	#Manages resource conflicts
	- locks
	#This stores policies and other information to not have to relearn everything every time.
	- log
	#Don't bother about anything else for the most part.

#This is where everything is being set up to be simulated and run.
- TestBed
	#These 3 files are critical. They set up and run everything.
	#This first one does it for simulated data. Unfortunately, it's not automated to run the paper experiments (they have to be done one by one). Will do this when I have more time!
	- testbed.py
	#This one does it for offline, real data. Same, not automated.
	- real_testbed.py
	#This does it for the live online experiments. This file contains all the code to post,
	#fetch data from MTurk while running one of the methods to change pricing. This is likely more useful if you downloaded this! You'd want to run our method (but you could also use this to run Dai's method which has basically been reimplemented by us). Gao's code was not reimplemented so we didn't release it.
	- live_online_experiment.py
	#The rest of the files should be left intact, they're automatically generated and managed by code
	#in live_online_experiment.py

---------------------------------------------------------------------------------------------------------------------------------------
cd into pymdptoolbox and run 
> python setup.py install

Troubleshooting
---------------------------------------------------------------------------------------------------------------------------------------
EM may give you errors since it's finicky; in that case just type make when you're in ..../QualityPOMDP/EM/
You'll need gsl for this.




