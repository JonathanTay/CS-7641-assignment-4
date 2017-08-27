'''
Created on Apr 9, 2017

@author: Jon
'''
import sys
sys.path.append('./burlap.jar')
import java
from collections import defaultdict
from time import clock
from burlap.behavior.policy import Policy;
from burlap.assignment4 import BasicGridWorld;
from burlap.behavior.singleagent import EpisodeAnalysis;
from burlap.behavior.singleagent.auxiliary import StateReachability;
from burlap.behavior.singleagent.auxiliary.valuefunctionvis import ValueFunctionVisualizerGUI;
from burlap.behavior.singleagent.learning.tdmethods import QLearning;
from burlap.behavior.singleagent.planning.stochastic.policyiteration import PolicyIteration;
from burlap.behavior.singleagent.planning.stochastic.valueiteration import ValueIteration;
from burlap.behavior.valuefunction import ValueFunction;
from burlap.domain.singleagent.gridworld import GridWorldDomain;
from burlap.oomdp.core import Domain;
from burlap.oomdp.core import TerminalFunction;
from burlap.oomdp.core.states import State;
from burlap.oomdp.singleagent import RewardFunction;
from burlap.oomdp.singleagent import SADomain;
from burlap.oomdp.singleagent.environment import SimulatedEnvironment;
from burlap.oomdp.statehashing import HashableStateFactory;
from burlap.oomdp.statehashing import SimpleHashableStateFactory;
from burlap.assignment4.util import MapPrinter;
from burlap.oomdp.core import TerminalFunction;
from burlap.oomdp.core.states import State;
from burlap.oomdp.singleagent import RewardFunction;
from burlap.oomdp.singleagent.explorer import VisualExplorer;
from burlap.oomdp.visualizer import Visualizer;
from burlap.assignment4.util import BasicRewardFunction;
from burlap.assignment4.util import BasicTerminalFunction;
from burlap.assignment4.util import MapPrinter;
from burlap.oomdp.core import TerminalFunction;
from burlap.assignment4.EasyGridWorldLauncher import visualizeInitialGridWorld
from burlap.assignment4.util.AnalysisRunner import calcRewardInEpisode, simpleValueFunctionVis,getAllStates
import csv
import os
from collections import deque
def dumpCSV(iters, times,rewards,steps,world,method,ow):
    
    fname = 'size {}.csv'.format(method)
    assert len(iters)== len(times)
    assert len(iters)== len(rewards)
    assert len(iters)== len(steps)
    if not os.path.exists(fname) or ow:
        with open(fname,'wb') as f: 
            f.write('iter,time,reward,steps,shape\n')
            
    with open(fname,'ab') as f:        
        writer = csv.writer(f,delimiter=',')
        writer.writerows(zip(iters,times,rewards,steps,world))
    
def runEvals(initialState,plan,rewardL,stepL):
    r = []
    s = []
    for trial in range(evalTrials):
        ea = plan.evaluateBehavior(initialState, rf, tf,4000);
        r.append(calcRewardInEpisode(ea))
        s.append(ea.numTimeSteps())
    rewardL.append(sum(r)/float(len(r)))
    stepL.append(sum(s)/float(len(s))) 
    
    



if __name__ == '__main__':
    
    discount=0.99
    MAX_ITERATIONS = 1000;
    NUM_INTERVALS = 1000;
    evalTrials = 100;
    for n in [2,3,4,5,7,10,15,20,25,30,35,40,50]:
#     n = 4
        #print n
        userMap = [[0]*n]*n
        world = '{}x{}'.format(n,n)
        
        tmp = java.lang.reflect.Array.newInstance(java.lang.Integer.TYPE,[n,n])
        for i in range(n):
            for j in range(n):
                tmp[i][j]= userMap[i][j]
        userMap = MapPrinter().mapToMatrix(tmp)
        maxX = maxY= n-1
        
        gen = BasicGridWorld(userMap,maxX,maxY)
        domain = gen.generateDomain()
        initialState = gen.getExampleState(domain);
    
        rf = BasicRewardFunction(maxX,maxY,userMap)
        tf = BasicTerminalFunction(maxX,maxY)
        env = SimulatedEnvironment(domain, rf, tf,initialState);
        #Print the map that is being analyzed
        print "/////Grid World {}x{} Analysis/////\n".format(n,n)
        MapPrinter().printMap(MapPrinter.matrixToMap(userMap));
    #     visualizeInitialGridWorld(domain, gen, env)
        hashingFactory = SimpleHashableStateFactory()
        increment = MAX_ITERATIONS/NUM_INTERVALS
        # Value Iteration
        iterations = defaultdict(list)
        timing = defaultdict(list)
        rewards = defaultdict(list)
        steps = defaultdict(list) 
        print "//Size Value Iteration Analysis//"        
        startTime = clock()
        vi = ValueIteration(domain,rf,tf,discount,hashingFactory,1e-6, MAX_ITERATIONS); 
        vi.toggleUseCachedTransitionDynamics(False)
            # run planning from our initial state
        vi.setDebugCode(0)
        p = vi.planFromState(initialState);
        timing['Value'].append(clock()-startTime)   
        iterations['Value'].append(vi.numIterations)       
        # evaluate the policy with one roll out visualize the trajectory
        runEvals(initialState,p,rewards['Value'],steps['Value'])
        
        MapPrinter.printPolicyMap(vi.getAllStates(), p, gen.getMap());
        print "\n\n"
    #     simpleValueFunctionVis(vi, p, initialState, domain, hashingFactory, "Value Iteration")

        dumpCSV(iterations['Value'], timing['Value'], rewards['Value'], steps['Value'], [n], 'Value',n ==2)
        
    # 
        print "//Size Policy Iteration Analysis//"            
        pi = PolicyIteration(domain,rf,tf,discount,hashingFactory,1e-6,20, MAX_ITERATIONS); #//Added a very high delta number in order to guarantee that value iteration occurs the max number of iterations for comparison with the other algorithms.
        pi.toggleUseCachedTransitionDynamics(False)
            # run planning from our initial state
        pi.setDebugCode(0)		
        pi.setPolicyToEvaluate(p)
        startTime = clock()
        p = pi.planFromState(initialState);
        timing['Policy'].append(clock()-startTime)  
        iterations['Policy'].append(pi.totalPolicyIterations)            
        # evaluate the policy with one roll out visualize the trajectory
        runEvals(initialState,p,rewards['Policy'],steps['Policy'])
        MapPrinter.printPolicyMap(pi.getAllStates(), p, gen.getMap());
        print "\n\n"
    #     simpleValueFunctionVis(pi, p, initialState, domain, hashingFactory, "Policy Iteration")
        dumpCSV(iterations['Policy'], timing['Policy'], rewards['Policy'], steps['Policy'], [n], 'Policy',n ==2)
        #continue
        lr=0.1
        epsilon=0.5
        qInit = 1
        last10Chg= deque([99]*10,maxlen=10)
        Qname = 'Q-Learning L{:0.1f} E{:0.1f}'.format(lr,epsilon)
        agent = QLearning(domain,discount,hashingFactory,qInit,lr,epsilon)
        agent.setDebugCode(0)
        print "//Size {} Iteration Analysis//".format(Qname)
        
        Qchg = 10.
        learningEpisodes = 0
        startTime = clock()        
        while Qchg > 1.0:           
            ea = agent.runLearningEpisode(env,4000)
            last10Chg.append(agent.maxQChangeInLastEpisode)
            learningEpisodes += 1
            Qchg = sum(last10Chg)/10.
            env.resetEnvironment()
        timing[Qname].append(clock()-startTime)
        agent.initializeForPlanning(rf, tf, 1)
        p = agent.planFromState(initialState)     # run planning from our initial state
        iterations[Qname].append(learningEpisodes)        
        # evaluate the policy with one roll out visualize the trajectory
        runEvals(initialState,p,rewards[Qname],steps[Qname])
        MapPrinter.printPolicyMap(getAllStates(domain,rf,tf,initialState), p, gen.getMap());
        print "\n\n\n\n"
    #             simpleValueFunctionVis(agent, p, initialState, domain, hashingFactory, Qname)
        dumpCSV(iterations[Qname], timing[Qname], rewards[Qname], steps[Qname], [n], Qname,n ==2)
#     