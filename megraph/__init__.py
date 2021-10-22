from megraph.egraph_constructor import Constructor
from megraph.language import downcast
from megraph.language import RecExprCompiler

def load_weights(relay_src):
    import tvm
    relay_code = open(relay_src, 'r').read()
    model = tvm.parser.fromtext(relay_code)
    shape_dict = dict()
    for args in model['main'].params:
        shape_dict[args.name_hint] = tuple(args.type_annotation.shape)
    return shape_dict

def load_egraph(f):
    with open(f, 'r') as fp:
        egraph_record = fp.read()
        egraph = Constructor().from_text(egraph_record)
        for _, eclass in egraph.eclasses.items():
            eclass.map(lambda x: downcast(x))
        return egraph