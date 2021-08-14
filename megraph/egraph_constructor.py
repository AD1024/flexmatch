from typing import List, Tuple
from megraph.eclass import *
from megraph.egraph import *
from megraph.language import *
from enum import Enum

class State(Enum):
    ACCEPT_INT = 0,
    ACCEPT_META = 1
    ACCEPT_BEG_ECLASS = 2
    ACCEPT_END_ECLASS = 3
    ACCEPT_BEG_ENODE = 4
    ACCEPT_END_ENODE = 5
    ACCEPT_BEG_SYM  = 6
    ACCEPT_END_SYM = 7
    ACCEPT_BEG_CHILD = 8
    ACCEPT_END_CHILD = 9
    ACCEPT_EID = 10
    ACCEPT_CHILDREN = 11
    ACCEPT_STR = 12
    ACCEPT_FIN = 13
    ERROR = 14
    END = 15

def accept_symbol_cmd(x):
    return x == 'BEGIN_SYMBOL'

def accept_meta(x):
    return x == 'SIZE'

def accept_end_symbol(x):
    return x == 'END_SYMBOL'

def accept_begin_children(x):
    return x == 'BEGIN_CHILDREN'

def accept_end_children(x):
    return x == 'END_CHILDREN'

def accept_begin_enode(x):
    return x == 'BEGIN_ENODE'

def accept_end_enode(x):
    return x == 'END_ENODE'

def accept_eclass(x):
    return x == 'ECLASS'

def accept_end_eclass(x):
    return x == 'END_ECLASS'

def accept_fin(x):
    return x == 'FIN'

def accept_str(x):
    return isinstance(x, str)

def accept_int(x):
    try:
        int(x)
        return True
    except:
        return False

def set_egraph_size(egraph: EGraph):
    def f(size):
        egraph.size = size
    return f

def set_op_eid(constructor):
    def f(eid):
        constructor.op_eid = eid
    return f

class Constructor:
    def __init__(self, instructions: List[str]):
        self.instructions = instructions
        self.state = set([State.ACCEPT_META])
        self.peek_state = None
        self.int_stack = []
        self.str_stack = []
        self.enode_stack = []
        self.egraph = EGraph()
        self.op_eid = 0
    
    def emit_enode(self):
        symbol = self.str_stack.pop()
        self.enode_stack.append(ENode(symbol, self.int_stack.copy()))
        self.int_stack.clear()

    def emit_eclass(self):
        eclass = EClass(self.op_eid, self.enode_stack.copy())
        self.enode_stack.clear()
        self.egraph[self.op_eid] = eclass

    def consume(self, token):
        if State.ERROR in self.state or State.END in self.state:
            raise Exception(f'Trying to consume a token while having State {self.state}')
        
        if State.ACCEPT_META in self.state and accept_meta(token):
            self.state = {State.ACCEPT_INT}
            self.peek_state = {State.ACCEPT_FIN, State.ACCEPT_BEG_ECLASS}
            self.resize = True
        elif State.ACCEPT_INT in self.state and accept_int(token):
            self.int_stack.append(int(token))
            if self.resize:
                self.egraph.size = self.int_stack.pop()
                self.egraph.eclasses = [None] * self.egraph.size
                self.resize = False
            if self.peek_state is not None:
                self.state = self.peek_state
                self.peek_state = None
        elif State.ACCEPT_BEG_ECLASS in self.state and accept_eclass(token):
            self.state = {State.ACCEPT_EID}
        elif State.ACCEPT_END_ECLASS in self.state and accept_end_eclass(token):
            self.emit_eclass()
            self.state = {State.ACCEPT_FIN, State.ACCEPT_BEG_ECLASS}
        elif State.ACCEPT_EID in self.state and accept_int(token):
            self.op_eid = int(token)
            self.state = {State.ACCEPT_END_ECLASS, State.ACCEPT_BEG_ENODE}
        elif State.ACCEPT_BEG_ENODE in self.state and accept_begin_enode(token):
            self.state = {State.ACCEPT_BEG_SYM, State.ACCEPT_BEG_CHILD}
        elif State.ACCEPT_END_ENODE in self.state and accept_end_enode(token):
            self.emit_enode()
            self.state = {State.ACCEPT_BEG_ENODE, State.ACCEPT_END_ECLASS}
        elif State.ACCEPT_BEG_SYM in self.state and accept_symbol_cmd(token):
            self.state = {State.ACCEPT_STR}
        elif State.ACCEPT_STR in self.state and accept_str(token):
            self.str_stack.append(token)
            self.state = {State.ACCEPT_END_SYM}
        elif State.ACCEPT_END_SYM in self.state and accept_end_symbol(token):
            self.state = {State.ACCEPT_BEG_CHILD, State.ACCEPT_END_ENODE}
        elif State.ACCEPT_BEG_CHILD in self.state and accept_begin_children(token):
            self.state = {State.ACCEPT_INT, State.ACCEPT_END_CHILD}
        elif State.ACCEPT_END_CHILD in self.state and accept_end_children(token):
            self.state = {State.ACCEPT_END_ENODE}
        elif State.ACCEPT_FIN in self.state and accept_fin(token):
            self.state = {State.END}
        else:
            self.state = {State.ERROR}
    
    def parse(self):
        for insn in self.instructions:
            self.consume(insn)
        return self.egraph