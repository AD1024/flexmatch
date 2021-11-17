import tvm
from tvm import relay

class RenameMutator(relay.ExprMutator):
        def __init__(self, rws):
            super().__init__()
            self.var_map = dict()
            self.rws = rws
        
        def visit_var(self, var):
            if var in self.var_map:
                return self.var_map[var]
            else:
                new_name = var.name_hint
                for (f, t) in self.rws.items():
                    new_name = new_name.replace(f, t)
                if new_name != var.name_hint:
                    new_var = relay.Var(new_name, type_annotation=var.type_annotation)
                    self.var_map[var] = new_var
                    return new_var
                else:
                    self.var_map[var] = var
                    return var

import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprMutator

class LetInliner(ExprMutator):
    def __init__(self):
        super().__init__()
        self.let_map = {}

    def visit_var(self, var):
        if var in self.let_map:
            return self.let_map[var]
        return var

    def visit_let(self, let):
        # don't deal with functions
        if isinstance(let.value, relay.Function):
            return super().visit_let(let)
        self.let_map[let.var] = self.visit(let.value)
        return self.visit(let.body)

class AlterDense(relay.ExprMutator):
    def __init__(self):
        super().__init__()
    
    def visit_call(self, call):
        args = [self.visit(x) for x in call.args]
        if call.op.name == 'nn.dense' and isinstance(args[0], relay.Constant):
            return relay.transpose(relay.Call(call.op, [args[1], args[0]], call.attrs, call.type_args, call.span))
        else:
            return relay.Call(call.op, args, call.attrs, call.type_args, call.span)

class RemoveAnnotations(relay.ExprMutator):
    def __init__(self):
        super().__init__()
    
    def visit_call(self, call):
        args = [self.visit(x) for x in call.args]
        if call.op.name == 'annotation.stop_fusion':
            return args[0]
        else:
            return relay.Call(call.op, args, call.attrs, call.type_args, call.span)