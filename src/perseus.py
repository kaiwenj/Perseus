import sys
sys.path.append('../../PBVI/src/')

import numpy as np
from PBVI import Backup, GetBetaA, GetBetaAO, BeliefTransition, argmaxAlpha
import time

class Perseus(object):
    
    def __init__(self, perseusBackup, getValue, convergenceTolerance, V):
        self.perseusBackup=perseusBackup
        self.getValue=getValue
        self.convergenceTolerance=convergenceTolerance
        self.V=V
        
    def __call__(self, B):
        delta=np.inf
        V=self.V
        vNew=[self.getValue(b, V) for b in B]
        while (delta > self.convergenceTolerance):
            v=vNew.copy()
            VNew=self.perseusBackup(B, V)
            vNew=[self.getValue(b, VNew) for b in B]
            delta=max([abs(vNew[i]-v[i]) for i in range(len(B))])
            V=VNew.copy()
        return V


class PerseusBackup(object):
    
    def __init__(self, selectBelief, updateB, updateV):
        self.selectBelief=selectBelief
        self.updateV=updateV
        self.updateB=updateB
        
    def __call__(self, B, V):
        VNew=[]
        BTilde=B.copy()
        while BTilde:
            b=self.selectBelief(BTilde)
            VNew=self.updateV(b, BTilde, VNew, V)
            BTildeNew=self.updateB(BTilde, VNew, V)
            BTilde=BTildeNew.copy()
        return VNew
            
    
class UpdateV(object):
    
    def __init__(self, backup, evaluateBelief, argmaxAlpha):
        self.backup=backup
        self.evaluateBelief=evaluateBelief
        self.argmaxAlpha=argmaxAlpha
        
    def __call__(self, b, B, VNew, V):
        alphaNew=self.backup(V, b)        
        alphaNewTimesBValue=self.evaluateBelief(b, alphaNew)
        alpha=self.argmaxAlpha(V, b)
        alphaTimesBValue=self.evaluateBelief(b, alpha)
        if alphaNewTimesBValue>=alphaTimesBValue:
            VNew.append(alphaNew)
        else:
            VNew.append(alpha)
        return VNew
   
     
class UpdateB(object):
    
    def __init__(self, getValue):
        self.getValue=getValue
        
    def __call__(self, B, VNew, V):
        BNew=[b for b in B if self.getValue(b, VNew)-self.getValue(b, V)<0]
        return BNew
        

class TigerTransition():
    def __init__(self):
        self.transitionMatrix = {
            ('listen', 'tiger-left', 'tiger-left'): 1.0,
            ('listen', 'tiger-left', 'tiger-right'): 0.0,
            ('listen', 'tiger-right', 'tiger-left'): 0.0,
            ('listen', 'tiger-right', 'tiger-right'): 1.0,

            ('open-left', 'tiger-left', 'tiger-left'): 0.5,
            ('open-left', 'tiger-left', 'tiger-right'): 0.5,
            ('open-left', 'tiger-right', 'tiger-left'): 0.5,
            ('open-left', 'tiger-right', 'tiger-right'): 0.5,

            ('open-right', 'tiger-left', 'tiger-left'): 0.5,
            ('open-right', 'tiger-left', 'tiger-right'): 0.5,
            ('open-right', 'tiger-right', 'tiger-left'): 0.5,
            ('open-right', 'tiger-right', 'tiger-right'): 0.5
        }

    def __call__(self, state, action, nextState):
        nextStateProb = self.transitionMatrix.get((action, state, nextState), 0.0)
        return nextStateProb


class TigerReward():
    def __init__(self, rewardParam):
        self.rewardMatrix = {
            ('listen', 'tiger-left'): rewardParam['listen_cost'],
            ('listen', 'tiger-right'): rewardParam['listen_cost'],

            ('open-left', 'tiger-left'): rewardParam['open_incorrect_cost'],
            ('open-left', 'tiger-right'): rewardParam['open_correct_reward'],

            ('open-right', 'tiger-left'): rewardParam['open_correct_reward'],
            ('open-right', 'tiger-right'): rewardParam['open_incorrect_cost']
        }

    def __call__(self, state, action, sPrime):
        rewardFixed = self.rewardMatrix.get((action, state), 0.0)
        return rewardFixed


class TigerObservation():
    def __init__(self, observationParam):
        self.observationMatrix = {
            ('listen', 'tiger-left', 'tiger-left'): observationParam['obs_correct_prob'],
            ('listen', 'tiger-left', 'tiger-right'): observationParam['obs_incorrect_prob'],
            ('listen', 'tiger-right', 'tiger-left'): observationParam['obs_incorrect_prob'],
            ('listen', 'tiger-right', 'tiger-right'): observationParam['obs_correct_prob'],

            ('open-left', 'tiger-left', 'Nothing'): 1,
            ('open-left', 'tiger-right', 'Nothing'): 1,
            ('open-right', 'tiger-left', 'Nothing'): 1,
            ('open-right', 'tiger-right', 'Nothing'): 1,
        }

    def __call__(self, state, action, observation):
        observationProb = self.observationMatrix.get((action, state, observation), 0.0)
        return observationProb



"""
def main():
    
    rewardParam={'listen_cost':-1, 'open_incorrect_cost':-100, 'open_correct_reward':10}
    rewardFunction=TigerReward(rewardParam)
    
    observationParam={'obs_correct_prob':0.85, 'obs_incorrect_prob':0.15}
    observationFunction=TigerObservation(observationParam)
    
    transitionFunction=TigerTransition()
    
    stateSpace=['tiger-left', 'tiger-right']
    observationSpace=['tiger-left', 'tiger-right', 'Nothing']
    actionSpace=['open-left', 'open-right', 'listen']
    
    beliefTransition=BeliefTransition(transitionFunction, observationFunction)
    getBetaAO=GetBetaAO(beliefTransition, argmaxAlpha)
    
    gamma=0.99
    roundingTolerance=5
    getBetaA=GetBetaA(getBetaAO, transitionFunction, rewardFunction, observationFunction, stateSpace, observationSpace, gamma, roundingTolerance)
    backup=Backup(getBetaA, argmaxAlpha, stateSpace, actionSpace)

    V=[{'action': 'listen', 'alpha':{s: min(rewardParam.values())/(1-gamma) for s in stateSpace}}]
    
    evaluateBelief=lambda b, alpha: sum([alpha['alpha'][s]*b[s] for s in b.keys()])
    updateV=UpdateV(backup, evaluateBelief, argmaxAlpha)
    
    getValue=lambda b, V: max([evaluateBelief(b, alpha) for alpha in V])
    updateB=UpdateB(getValue)
    
    selectBelief=lambda B: B[np.random.choice(len(B))]
    perseusBackup=PerseusBackup(selectBelief, updateB, updateV)
    
    convergenceTolerance=1e-5
    perseus=Perseus(perseusBackup, getValue, convergenceTolerance, V)
    
    B=[{'tiger-left':0.05*n, 'tiger-right':1-0.05*n} for n in range(21)]
    VNew=perseus(B)
    print(VNew)
    
    a=[argmaxAlpha(VNew, b)['action'] for b in B]
    print(a)
"""
def transitionFunction(s, a, sPrime):
    if s=='GameOver':
        return 1*(sPrime=='GameOver')
    if a in ['shootUp', 'shootDown']:
        return 1*(sPrime=='GameOver')
    if a=='stay':
        return 1*(sPrime==s)
    if a=='move':
        if sPrime=='GameOver':
            return 0
        agentLocation, wumpusLocation=s
        agentLocationPrime, wumpusLocationPrime=sPrime
        return (agentLocation=='Left' and agentLocationPrime=='Right' and wumpusLocation==wumpusLocationPrime)+\
               (agentLocation=='Right' and agentLocationPrime=='Left' and wumpusLocation==wumpusLocationPrime)
    
def rewardFunctionFull(s, a, sPrime, movingCost, victoryReward, losingPenalty):
    if s=='GameOver':
        return 0
    if a not in ['shootUp', 'shootDown']:
        return movingCost
    agentLocation, wumpusLocation=s
    if a=='shootUp':
        return (wumpusLocation=='Up')*victoryReward+(wumpusLocation!='Up')*losingPenalty
    if a=='shootDown':
        if agentLocation=='Left':
            return (wumpusLocation=='BottomLeft')*victoryReward+(wumpusLocation!='BottomLeft')*losingPenalty
        if agentLocation=='Right':
            return (wumpusLocation=='BottomRight')*victoryReward+(wumpusLocation!='BottomRight')*losingPenalty
        

def observationFunctionFull(sPrime, a, o, observationAccuracy):
    if sPrime=='GameOver':
        return 1*(o==('',))
    agentLocation, wumpusLocation=sPrime
    if agentLocation=='Left':
        if wumpusLocation in ['Up','BottomLeft']:
            return observationAccuracy*(o==('Stench',))+(1-observationAccuracy)*(o==('',))
        else:
            return (1-observationAccuracy)*(o==('Stench',))+observationAccuracy*(o==('',))
    if agentLocation=='Right':
        if wumpusLocation in ['Up','BottomRight']:
            return observationAccuracy*(o==('Stench',))+(1-observationAccuracy)*(o==('',))
        else:
            return (1-observationAccuracy)*(o==('Stench',))+observationAccuracy*(o==('',))
    return 0

def main():
    
    stateSpace=[(agentLocation, wumpusLocation) for agentLocation in ['Left', 'Right'] for wumpusLocation in ['Up', 'BottomLeft', 'BottomRight']]+['GameOver']
    actionSpace=['stay', 'move', 'shootUp', 'shootDown']
    observationSpace=[('Stench',), ('',)]
    
    movingCost=-1
    victoryReward=100
    losingPenalty=-100
    rewardFunction=lambda s, a, sPrime: rewardFunctionFull(s, a, sPrime, movingCost, victoryReward, losingPenalty)
    
    observationAccuracy=0.85
    observationFunction=lambda sPrime, a, o: observationFunctionFull(sPrime, a, o, observationAccuracy)
    
    #simulatePOMDP=SimulatePOMDP(stateSpace, actionSpace, observationSpace, transitionFunction, rewardFunction, observationFunction)

    beliefTransition=BeliefTransition(transitionFunction, observationFunction)
    getBetaAO=GetBetaAO(beliefTransition, argmaxAlpha)
    
    gamma=0.95
    roundingTolerance=5
    getBetaA=GetBetaA(getBetaAO, transitionFunction, rewardFunction, observationFunction, stateSpace, observationSpace, gamma, roundingTolerance)
    backup=Backup(getBetaA, argmaxAlpha, stateSpace, actionSpace)

    worstQ=movingCost/(1-gamma)+losingPenalty
    V=[{'action': 'MoveVertical', 'alpha':{state: worstQ for state in stateSpace}}]
    
    evaluateBelief=lambda b, alpha: sum([alpha['alpha'][s]*b[s] for s in b.keys()])
    updateV=UpdateV(backup, evaluateBelief, argmaxAlpha)
    
    getValue=lambda b, V: max([evaluateBelief(b, alpha) for alpha in V])
    updateB=UpdateB(getValue)
    
    selectBelief=lambda B: B[np.random.choice(len(B))]
    perseusBackup=PerseusBackup(selectBelief, updateB, updateV)
    
    convergenceTolerance=1e-5
    perseus=Perseus(perseusBackup, getValue, convergenceTolerance, V)
    
    b0={('Left', 'Up'): 1/3,
        ('Left', 'BottomLeft'): 1/3,
        ('Left', 'BottomRight'): 1/3, 
        ('Right', 'Up'): 0,
        ('Right', 'BottomLeft'): 0,
        ('Right', 'BottomRight'): 0,
        'GameOver': 0}
    
    
    grid=[0.05*x for x in range(21)]
    probability=[(x, y, z) for x in grid for y in grid for z in grid if x+y+z==1]
    
    
    B=[{('Left', 'Up'): x,
        ('Left', 'BottomLeft'): y,
        ('Left', 'BottomRight'): z, 
        ('Right', 'Up'): 0,
        ('Right', 'BottomLeft'): 0,
        ('Right', 'BottomRight'): 0,
        'GameOver': 0} for x, y, z in probability]+[{('Left', 'Up'): 0,
        ('Left', 'BottomLeft'): 0,
        ('Left', 'BottomRight'): 0, 
        ('Right', 'Up'): x,
        ('Right', 'BottomLeft'): y,
        ('Right', 'BottomRight'): z,
        'GameOver': 0} for x, y, z in probability]
    
    t=time.time()
    VNew=perseus(B)
    elapsed = time.time() - t
    print(elapsed)
    a=[argmaxAlpha(VNew, b)['action'] for b in B]
    
                                                     
                                                     
    

if __name__=="__main__":
    main()
