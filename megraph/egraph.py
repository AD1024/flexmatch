from megraph.eclass import EClass
from tvm
from typing import List

class EGraph:
    def __init__(self, size: int):
        self.eclasses = [None] * size
        self.size = size
    
    def __getitem__(self, index) -> EClass:
        assert index < self.size
        return self.eclasses[index]

    def add(self, eclass: EClass):
        assert eclass.eid < self.size
        self.eclasses[eclass.eid] = eclass

class EGraphMatcher:
    '''
    Given an EGraph and a relay program, try to do exahustive
    matching on the program. The given EGraph might represent
    some program fragments that can be matched against
    a part of the relay program.
    '''
    def __init__(self, egraphs=[], accepts=[], lang_to_symbol=lambda *_: None, match_symbol=lambda *_: False):
        self.egraph: List[EGraph] = egraphs
        self.accepts: set = set(accepts)
        self.l2s = lang_to_symbol
        self.match_symbol = match_symbol    
