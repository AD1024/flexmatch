from megraph.egraph_constructor import Constructor

instructions = [
    'SIZE', '4',
    'ECLASS', '0',
    'BEGIN_ENODE', 'BEGIN_SYMBOL', 'x', 'END_SYMBOL', 'BEGIN_CHILDREN', '1', '2', 'END_CHILDREN', 'END_ENODE',
    'BEGIN_ENODE', 'BEGIN_SYMBOL', '<<', 'END_SYMBOL', 'BEGIN_CHILDREN', '3', 'END_CHILDREN', 'END_ENODE',
    'END_ECLASS',
    'ECLASS', '1',
    'BEGIN_ENODE', 'BEGIN_SYMBOL', 'a', 'END_SYMBOL', 'END_ENODE',
    'END_ECLASS',
    'ECLASS', '2',
    'BEGIN_ENODE', 'BEGIN_SYMBOL', '2', 'END_SYMBOL', 'END_ENODE',
    'BEGIN_ENODE', 'BEGIN_SYMBOL', '+', 'END_SYMBOL', 'BEGIN_CHILDREN', '3', '3', 'END_CHILDREN', 'END_ENODE',
    'END_ECLASS',
    'ECLASS', '3',
    'BEGIN_ENODE', 'BEGIN_SYMBOL', '1', 'END_SYMBOL', 'END_ENODE',
    'END_ECLASS',
]

constructor = Constructor(instructions)
print(constructor.parse())