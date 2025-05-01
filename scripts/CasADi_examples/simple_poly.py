import casadi as cas
from casadi.tools.structure3 import struct_symSX, struct_symMX


class _SymVar:
    def __init__(self, symvar_type):
        assert symvar_type in ['SX', 'MX'], 'symvar_type must be either SX or MX, you have: {}'.format(symvar_type)

        if symvar_type == 'MX':
            self.sym = cas.MX.sym
            self.sym_struct = struct_symMX
            self.dtype = cas.MX
        if symvar_type == 'SX':
            self.sym = cas.SX.sym
            self.sym_struct = struct_symSX
            self.dtype = cas.SX


sv = _SymVar('SX')

_x =   {'name': ['Objective Variable'], 'var': [sv.sym('w'), (0,0)]}


print(f"x : {_x}")
# Extract the variable
w = _x['var'][0]
print(w)

obj = w**2 - 6*w + 13

print(f"obj:{obj}")

nlp = {'x': w, 'f': obj}

solver = cas.nlpsol('solver', 'ipopt', nlp)

sol = solver(x0 = 0)

# Extract solution
w_opt = sol['x']
f_opt = sol['f']

# Output
print(f"Optimal w: {w_opt}")
print(f"Minimum value of objective: {f_opt}")
