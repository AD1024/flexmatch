from functools import reduce
from megraph.language import *
from tvm import relay
from tvm.relay.expr import Expr
from tvm.relay.expr_functor import ExprFunctor
from typing import List
from megraph.eclass import EClass
from megraph.from_relay import convert_relay_op

class EGraph:
    def __init__(self, size: int=0):
        self.eclasses = dict()
        self.size = size
    
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

class EGraphMatcher(ExprFunctor):
    '''
    Given an EGraph and a relay program, try to do exahustive
    matching on the program. The given EGraph might represent
    some program fragments that can be matched against
    a part of the relay program.
    '''
    def __init__(self, egraphs=[], accept=0):
        self.egraph: List[EGraph] = egraphs
        self.accept: int = accept
        self.matched_vars: dict = dict()
        self.var_binding: dict = dict()
    
    def visit_var(self, var):
        if var in self.var_binding:
            return self.visit(self.var_binding[var])
        else:
            eclass = self.egraph[self.accept]
            for enode in eclass.children:
                if isinstance(enode, Symbol):
                    self.matched_vars[var] = enode.symbol
                    return True
            return False

    def visit_call(self, call: relay.Call):
        op = convert_relay_op(call.op)
        eclass = self.egraph[self.accept]
        for enode in eclass.children:
            if isinstance(enode, RelayOperatorCall):
                if enode.children[0] == op and len(enode.children) - 1 == len(call.args):
                    current_accept = self.accept
                    for (ch_eid, expr) in zip(enode.children[1:], call.args):
                        self.accept = ch_eid
                        if not self.visit(expr):
                            self.accept = current_accept
                            return False
                    return True
                else:
                    return False
        return False