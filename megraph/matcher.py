from megraph.eclass import EClass
from tvm import relay
from typing import Tuple
from tvm.relay.expr_functor import ExprFunctor, ExprMutator
from tvm.relay.analysis import free_vars
from megraph.language import *
from megraph.from_relay import convert_relay_op
from megraph.egraph import EGraph

EG = 0
COMPILER = 1
SYMBOL = 2

class EGraphMatcher(ExprFunctor):
    '''
    Given an EGraph and a relay program, try to do exahustive
    matching on the program. The given EGraph might represent
    some program fragments that can be matched against
    a part of the relay program.
    '''
    def __init__(self, rw: Tuple[EGraph, str, str], accept=0):
        super().__init__()
        # An rw is a tuple of an EGraph
        # the compiler, the symbol and the variable matching between variables in
        # the patterns and the variabels mentioned in accelerator call API
        self.rw: Tuple[EGraph, str, str] = rw
        self.accept: int = self.rw[EG].root
        # Vars in the model -> Vars in EGraph
        self.matched_vars: dict = dict()
        # Vars in EGraph -> Expr in the model
        self.binding: dict = dict()
    
    def reset(self):
        self.accept = self.rw[EG].root
        self.matched_vars.clear()
        self.binding.clear()
    
    def visit_var(self, var):
        eclass = self.rw[EG][self.accept]
        if var in self.matched_vars:
            for enode in eclass.nodes:
                if isinstance(enode, Symbol) and self.matched_vars[var] == enode:
                    return True
            return False
        else:
            for enode in eclass.nodes:
                if isinstance(enode, Symbol):
                    self.matched_vars[var] = enode
                    return True
            return False

    def visit_call(self, call: relay.Call):
        eclass: EClass = self.rw[EG][self.accept]
        # Check whether there are equivalent operator calls represented
        # by a (sub)AST of the current accept eclass
        current_accept = self.accept
        for enode in filter(lambda enode: isinstance(enode, RelayOperatorCall), eclass.nodes):
            if convert_relay_op(call.op) == enode.symbol:
                print(call.op, enode)
                for (ch_eid, expr) in zip(enode.children, call.args):
                    self.accept = ch_eid
                    if not self.visit(expr):
                        self.accept = current_accept
                        return False
                return True
        # Check whether there are variables in current eclass bound to
        # an equivalent operator call
        for enode in filter(lambda enode: isinstance(enode, Symbol), eclass.nodes):
            if enode.symbol in self.binding and tvm.ir.structural_equal(self.binding[enode.symbol], call):
                return True
        # Otherwise bind an available variable to the current operator call
        # there should be only exactly 1 variable in this eclass, or the previous
        # matching should return True
        for enode in filter(lambda enode: isinstance(enode, Symbol) and enode.symbol not in self.binding, eclass.nodes):
            self.binding[enode] = call
            return True
        return False
    
    def visit(self, expr):
        if not isinstance(expr, relay.Call) and not isinstance(expr, relay.Var):
            return False
        else:
            return super().visit(expr)

def deduplicate_vars(expr):
    """
    Given the expr, replace all vars in the expression with fresh ones.
    This is done to preserve well-formedness in Relay (all var definitions must be unique)
    """
    class Deduplicator(ExprMutator):
        def __init__(self):
            super().__init__()
            self.var_map = {}

        def visit_var(self, var):
            if var in self.var_map:
                return self.var_map[var]
            fresh_var = relay.Var(var.name_hint, type_annotation=var.type_annotation)
            self.var_map[var] = fresh_var
            return fresh_var

        def visit_pattern(self, pattern):
            if isinstance(pattern, relay.PatternWildcard):
                return pattern
            if isinstance(pattern, relay.PatternVar):
                return relay.PatternVar(self.visit(pattern.var))
            if isinstance(pattern, relay.PatternTuple):
                return relay.PatternTuple([self.visit_pattern(subpattern)
                                           for subpattern in pattern.patterns])
            if isinstance(pattern, relay.PatternConstructor):
                return relay.PatternConstructor(pattern.constructor,
                                                [self.visit_pattern(subpattern)
                                                 for subpattern in pattern.patterns])
            raise ValueError(f"Invalid pattern {pattern}")

        def visit_match(self, match):
            new_val = self.visit(match.data)
            clauses = [relay.Clause(self.visit_pattern(c.lhs), self.visit(c.rhs))
                       for c in match.clauses]
            return relay.Match(new_val, clauses)

    dedup = Deduplicator()
    return dedup.visit(expr)

def conditional_anf(expr: relay.Expr, bindings: dict) -> relay.Expr:
    class Mutator(relay.ExprMutator):
        def __init__(self):
            super().__init__()
            self.bindings: dict = dict(list(map(lambda x: (x[1], x[0]), bindings.items())))
            self.var_map: dict = dict(list(map(lambda v: (v, relay.var(v.symbol)), self.bindings.values())))
        
        def visit(self, e):
            for (k, v) in self.bindings.items():
                if tvm.ir.structural_equal(k, e):
                    return self.var_map[v]
            return super().visit(e)
    
    mutator = Mutator()
    result = mutator.visit(expr)
    return (result, mutator.var_map)

def check_and_annotate(expr, rw: Tuple[EGraph, str, str]):
    egraph_matcher = EGraphMatcher(rw)
    class Matcher(relay.ExprMutator):
        def __init__(self):
            super().__init__()
            self.compiler_name = rw[COMPILER]
            self.composite_name = rw[SYMBOL]
            self.composite_counter = 0
        
        def extract_target(self, expr):
            mut_expr, var_mapping = conditional_anf(expr, egraph_matcher.binding)
            fv = free_vars(mut_expr)
            matched_vars = egraph_matcher.matched_vars.copy()
            matched_vars.update(map(lambda x: (x[1], x[0]), var_mapping.items()))
            # Ensure all variables in the lifted expression
            # are captured by the pattern
            assert(all(map(lambda x: x in matched_vars, fv)))
            inner_body = deduplicate_vars(mut_expr)
            inner_args = free_vars(inner_body)
            inner_func = relay.Function(inner_args, inner_body)

            inner_func = inner_func.with_attr("Composite", self.composite_name)

            outer_args = [relay.Var(f"outer_arg_{i}") for i in range(len(inner_args))]
            outer_func = relay.Function(outer_args, inner_func(*outer_args))
            outer_func = outer_func.with_attr("Compiler", self.compiler_name)
            outer_func = outer_func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
            outer_func = outer_func.with_attr(
                "global_symbol",
                f"{self.composite_name}_{self.composite_counter}")
            self.composite_counter += 1
            def construct_let(bindings, var_mapping):
                if len(bindings) == 0:
                    return outer_func(*fv)
                else:
                    (enode, expr), *xs = bindings
                    relay_var = var_mapping[enode]
                    return relay.Let(relay_var, expr, construct_let(xs, var_mapping))
            return construct_let(egraph_matcher.binding.items(), var_mapping)

        def visit(self, expr):
            if egraph_matcher.visit(expr):
                return self.extract_target(expr)
            else:
                egraph_matcher.reset()
                return super().visit(expr)
    return Matcher().visit(expr)
