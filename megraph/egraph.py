from megraph.eclass import EClass
from megraph.language import *

class EGraph:
    def __init__(self, size: int=0):
        self.eclasses = dict()
        self.size = size
        self.root = 0
    
    def __getitem__(self, index) -> EClass:
        assert index in self.eclasses
        return self.eclasses[index]
    
    def __setitem__(self, index, item):
        assert isinstance(item, EClass)
        self.eclasses[index] = item

    def add(self, eclass: EClass):
        self.eclasses[eclass.eid] = eclass
    
    def __str__(self):
        return f'{self.eclasses}'
    
    def __repr__(self):
        return self.__str__()

def sanitize_egraph(egraph: EGraph, var_mapping: dict):
    '''
    Checks whether a pattern represented in the EGraph
    does not capture all variables required by the accelerator call
    '''
    symbolset_memo = dict()
    def dfs(eid: int, enode: ENode):
        if enode in symbolset_memo:
            return symbolset_memo[(eid, enode)]
        if isinstance(enode, Symbol):
            return { enode.symbol }
        else:
            symbols = set()
            for eid in enode.children:
                eclass: EClass = egraph[eid]
                for ch in eclass.nodes:
                    symbols = symbols.union(dfs(eid, ch))