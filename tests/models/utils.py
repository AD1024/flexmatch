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