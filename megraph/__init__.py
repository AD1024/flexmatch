from megraph.egraph_constructor import Constructor
from megraph.language import downcast
from megraph.language import RecExprCompiler

def load_egraph(f):
    with open(f, 'r') as fp:
        egraph_record = fp.read()
        egraph = Constructor().from_text(egraph_record)
        for _, eclass in egraph.eclasses.items():
            eclass.map(lambda x: downcast(x))
        return egraph