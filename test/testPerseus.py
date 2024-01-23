
import sys
sys.path.append('../src/')
import numpy as np

import unittest
from ddt import ddt, data, unpack
import perseus as targetCode

@ddt
class TestPerseusBackup(unittest.TestCase):
    
    @data((lambda B: B[0], lambda b, B, VNew, V: V, lambda B, VNew, V: [b-1 for b in B if b>V[0]], [1, 2, 3, 4, 5], [2, 3], [2, 3]))
    @unpack
    def testPerseusBackupOnlyChangeB(self, selectBelief, updateV, updateB, B, V, expectedResult):
        perseusBackup=targetCode.PerseusBackup(selectBelief, updateB, updateV)
        calculatedResult=perseusBackup(B, V)
        self.assertListEqual(calculatedResult, expectedResult)
        
    @data((lambda B: B[0], lambda b, B, VNew, V: [v+1 for v in V], lambda B, VNew, V: [b-1 for b in B if b>V[0]], [1, 2, 3, 4, 5], [2, 3], [3, 4]))
    @unpack
    def testPerseusBackupVChanges(self, selectBelief, updateV, updateB, B, V, expectedResult):
        perseusBackup=targetCode.PerseusBackup(selectBelief, updateB, updateV)
        calculatedResult=perseusBackup(B, V)
        self.assertListEqual(calculatedResult, expectedResult)
        
    @data((lambda B: B[0], lambda b, B, VNew, V: [v+1 for v in V], lambda B, VNew, V: [b-1 for b in B if b>V[0]], [1, 2, 3, 4, 5], [1, 2], [2, 3]))
    @unpack
    def testPerseusBackupVNewDoesNotChange(self, selectBelief, updateV, updateB, B, V, expectedResult):
        perseusBackup=targetCode.PerseusBackup(selectBelief, updateB, updateV)
        calculatedResult=perseusBackup(B, V)
        self.assertListEqual(calculatedResult, expectedResult)
        
    @data((lambda B: B[0], lambda b, B, VNew, V: [B[0]], lambda B, VNew, V: [b-2 for b in B if b>VNew[0]], [1, 2, 3, 4, 5], [1, 2], [-3]))
    @unpack
    def testPerseusBackupVNewBSynergize(self, selectBelief, updateV, updateB, B, V, expectedResult):
        perseusBackup=targetCode.PerseusBackup(selectBelief, updateB, updateV)
        calculatedResult=perseusBackup(B, V)
        self.assertListEqual(calculatedResult, expectedResult)
        
    @data((lambda B: min(B), lambda b, B, VNew, V: [b], lambda B, VNew, V: [b-2 for b in B if b>VNew[0]], [5, 4, 3, 2, 1], [1, 2], [-3]))
    @unpack
    def testPerseusBackupVNewBSelection(self, selectBelief, updateV, updateB, B, V, expectedResult):
        perseusBackup=targetCode.PerseusBackup(selectBelief, updateB, updateV)
        calculatedResult=perseusBackup(B, V)
        self.assertListEqual(calculatedResult, expectedResult)
        
    def tearDown(self):
        pass


@ddt
class TestUpdateV(unittest.TestCase):
    
    @data((lambda V, b: V[0], lambda b, alpha: sum(alpha)+b, lambda V, b: V[-1], 
           3, [1, 2, 3, 4, 5], [], [[1, 2, 3], [6, 7, 8], [4, 7, 5], [6, 7, 7], [3, 2, 1]], [[1, 2, 3]]))
    @unpack
    def testUpdateVbNotInvolved(self, backup, evaluateBelief, argmaxAlpha, b, B, VNew, V, expectedResult):
        updateV=targetCode.UpdateV(backup, evaluateBelief, argmaxAlpha)
        calculatedResult=updateV(b, B, VNew, V)
        self.assertListEqual(calculatedResult, expectedResult)
        
    @data((lambda V, b: V[-1], lambda b, alpha: sum(alpha), lambda V, b: V[1], 
           3, [1, 2, 3, 4, 5], [[1, 2, 3]], [[1, 2, 3], [6, 7, 8], [4, 7, 5], [6, 7, 7], [3, 2, 1]], [[1, 2, 3], [6, 7, 8]]))
    @unpack
    def testUpdateAddAlphaAlreadyExist(self, backup, evaluateBelief, argmaxAlpha, b, B, VNew, V, expectedResult):
        updateV=targetCode.UpdateV(backup, evaluateBelief, argmaxAlpha)
        calculatedResult=updateV(b, B, VNew, V)
        self.assertListEqual(calculatedResult, expectedResult)
        
    @data((lambda V, b: [v+10 for v in V[0]], lambda b, alpha: sum(alpha), lambda V, b: V[1], 
           3, [1, 2, 3, 4, 5], [[1, 2, 3]], [[1, 2, 3], [6, 7, 8], [4, 7, 5], [6, 7, 7], [3, 2, 1]], [[1, 2, 3], [11, 12, 13]]))
    @unpack
    def testUpdateAddNewAlpha(self, backup, evaluateBelief, argmaxAlpha, b, B, VNew, V, expectedResult):
        updateV=targetCode.UpdateV(backup, evaluateBelief, argmaxAlpha)
        calculatedResult=updateV(b, B, VNew, V)
        self.assertListEqual(calculatedResult, expectedResult)
        
    @data((lambda V, b: [b+10 for v in V[0]], lambda b, alpha: alpha[0]*b-alpha[1]/b, lambda V, b: V[1], 
           5, [1, 2, 3, 4, 5], [[1, 2, 3]], [[1, 2, 3], [6, 7, 8], [4, 7, 5], [6, 7, 7], [3, 2, 1]], [[1, 2, 3], [15, 15, 15]]))
    @unpack
    def testUpdateInvolveb(self, backup, evaluateBelief, argmaxAlpha, b, B, VNew, V, expectedResult):
        updateV=targetCode.UpdateV(backup, evaluateBelief, argmaxAlpha)
        calculatedResult=updateV(b, B, VNew, V)
        self.assertListEqual(calculatedResult, expectedResult)
        
    def tearDown(self):
        pass
    

@ddt
class TestUpdateB(unittest.TestCase):
    
    @data((lambda b, V: V[0][1], [1, 2, 3, 4, 5], [[1, 2, 3]], [[1, 2, 3], [6, 7, 8], [4, 7, 5], [6, 7, 7], [3, 2, 1]], []))
    @unpack
    def testUpdateBNoneIncluded(self, getValue, B, VNew, V, expectedResult):
        updateB=targetCode.UpdateB(getValue)
        calculatedResult=updateB(B, VNew, V)
        self.assertListEqual(calculatedResult, expectedResult)
        
    @data((lambda b, V: b+max([v[1] for v in V]), [1, 2, 3, 4, 5], [[1, 2, 3]], [[1, 2, 3], [6, 7, 8], [4, 7, 5], [6, 7, 7], [3, 2, 1]], 
           [1, 2, 3, 4, 5]))
    @unpack
    def testUpdateBAllIncluded(self, getValue, B, VNew, V, expectedResult):
        updateB=targetCode.UpdateB(getValue)
        calculatedResult=updateB(B, VNew, V)
        self.assertListEqual(calculatedResult, expectedResult)
        
    @data((lambda b, V: V[b-1][0], [1, 2, 3, 4, 5], [[1, 2, 3], [7, 7, 8], [2, 7, 8], [6, 7, 8], [1, 7, 8]],
          [[1, 2, 3], [6, 7, 8], [4, 7, 5], [6, 7, 7], [3, 2, 1]], [3, 5]))
    @unpack
    def testUpdateBPartiallyIncluded(self, getValue, B, VNew, V, expectedResult):
        updateB=targetCode.UpdateB(getValue)
        calculatedResult=updateB(B, VNew, V)
        self.assertListEqual(calculatedResult, expectedResult)
            
    def tearDown(self):
        pass


@ddt
class TestPerseus(unittest.TestCase):
    
    @data((lambda B, V: V+[V[-1]/10], lambda b, V: b*V[-1], 1, [1, 2, 3], [3, 5, 7], [1, 2, 3, 0.3, 0.03, 0.003]))
    @unpack
    def testPerseusAdd(self, perseusBackup, getValue, convergenceTolerance, V, B, expectedResult):
        perseus=targetCode.Perseus(perseusBackup, getValue, convergenceTolerance, V)
        calculatedResult=perseus(B)
        self.assertListEqual(calculatedResult, expectedResult)
        
    @data((lambda B, V: [v/2 for v in V], lambda b, V: b*V[0], 2, [1, 2, 3], [2, 4, 6], [0.25, 0.5, 0.75]))
    @unpack
    def testPerseusChange(self, perseusBackup, getValue, convergenceTolerance, V, B, expectedResult):
        perseus=targetCode.Perseus(perseusBackup, getValue, convergenceTolerance, V)
        calculatedResult=perseus(B)
        self.assertListEqual(calculatedResult, expectedResult)
        
            
    def tearDown(self):
        pass




if __name__ == '__main__':
	unittest.main(verbosity=2)






