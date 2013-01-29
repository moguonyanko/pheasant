#!/usr/bin/python3
# -*- coding: utf-8 -*-

#Reference: 実用CommonLisp（Peter Norvig）

import pheasant.util as ut

def natp(n):
	'''
	Is n Natural Number?.
	'''
	if n < 0: return False
	elif n == 0: return True
	else: return natp(n-1)

'''
Indicates pattern match failure.
'''
FAIL = None

'''
Indicates pattern match success, with no valiables.
'''
NO_BINDINGS = (True, True)

class Symbol(object):
	'''
	Symbol type class as that in Lisp.
	'''
	def __init__(self, value):
		'''
		value: Include value in Symbol.
		'''
		self.value = value

	def __setattr__(self, key, value):
		'''
		Symbol value is constant.
		Raise syntax error.
		'''
		if hasattr(self, key):
			raise AttributeError("Symbol value is constant.")
		else:
			object.__setattr__(self, key, value)

def is_symbol(x):
	'''
	Is x a Symbol object?
	'''
	return isinstance(x, str)
	#return isinstance(x, Symbol)

def is_variable(x):
	'''
	IS x a variable (a symbol begining with `?')?
	'''
	return is_symbol(x) and x[0] == "?"

def binding_var(binding):
	'''
	Get the variable part of a single binding.
	'''
	return binding[0]

def binding_val(binding):
	'''
	Get the value part of a single binding.
	'''
	return binding[1]

def get_binding(var, bindings):
	'''
	Find a (variable . value) pair in a binding list.
	'''
	return ut.assoc(var, bindings)

def make_binding(var, val):
	'''
	Make tuple as binding.
	'''
	return (var, val)

def lookup(var, bindings):
	'''
	Get the value part (for var) from a binding list.
	'''
	return binding_val(get_binding(var, bindings))

def extend_bindings(var, val, bindings):
	'''
	Add a (var . value) pair to a binding list.
	'''
	part = bindings	if bindings != NO_BINDINGS else None
		
	return tuple(make_bindings(var, val), part)

def match_variable(var, inpt, bindings):
	'''
	Does VAR match input? Uses (or updates) and returns bindings.
	'''
	binding = get_binding(var, bindings)
	if binding == None:
		return extend_bindings(var, inpt, bindings)
	elif inpt == binding_val(binding):
		return bindings
	else:
		return FAIL

def is_cons(x):
	'''
	Is tuple as cons?
	'''
	return isinstance(x, tuple)

def first(x):
	'''
	Get first value at sequence.
	'''
	return x[0]

def rest(x):
	'''
	Get rest values at sequence.
	'''
	return x[1:]	

def unify(x, y, bindings=NO_BINDINGS):
	'''
	See if x and y match with given bindings.
	'''
	if bindings == FAIL:
		return FAIL
	elif is_valiable(x):
		return unify_variable(x, y, bindings)
	elif is_valiable(y):
		return unify_variable(y, x, bindings)
	elif x == y:
		return bindings
	elif is_cons(x) and is_cons(y):
		return	unify(rest(x), rest(y), (unify(first(x), first(y), bindings)))
	else:
	 return FAIL

def unify_valiables(var, x, bindings):
	'''
	Unify var with x, using (and maybe extending) bindings.
	'''
	if get_binding(var, bindings) != None:
		return unify(lookup(var, bindings), x, bindings)
	else:
		return extend_bindings(var, x, bindings)
	
if __name__ == '__main__':
	print("logic module load")


