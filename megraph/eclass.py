class ENode:
    def __init__(self, symbol, node_type):
        self.children = []  # EClass Ids
        self.parent = None
        self.symbol = symbol
        self.node_type = node_type
    
    def canonicalize(self):
        if self.parent is not None:
            return self.parent.canonicalize()

class EClass:
    def __init__(self, size, eid):
        self.nodes = [None] * size  # List[ENode]
        self.parent = None
        self.eid = eid
        self.size = size