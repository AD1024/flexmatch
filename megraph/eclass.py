class ENode:
    def __init__(self, symbol, children=[]):
        self.children = children  # EClass Ids
        self.parent = None
        self.symbol = symbol
    
    def canonicalize(self):
        if self.parent is not None:
            return self.parent.canonicalize()
        return self
    
    def map_children(self, f):
        list(map(f, self.children))
    
    def __str__(self):
        return f'({self.symbol} {self.children})'

    def __repr__(self):
        return self.__str__()

class EClass:
    def __init__(self, eid, nodes=[]):
        self.nodes = nodes  # List[ENode]
        self.parent = None
        self.eid = eid
        self.size = len(nodes)
    
    def for_each(self, f):
        list(map(f, self.nodes))
    
    def map(self, f):
        self.nodes = list(map(f, self.nodes))
    
    def __iter__(self):
        return iter(self.nodes)
    
    def __str__(self):
        return f'(EClass {self.eid} {self.nodes})'
    
    def __repr__(self):
        return self.__str__()