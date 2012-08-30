#!/usr/bin/python3
# -*- coding: utf-8 -*-

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

def natp(n):
	'''
	Is n Natural Number?.
	'''
	if n < 0: return False
	elif n == 0: return True
	else: return natp(n-1)

def unify(u, v):
	'''
	Unification substitution.
	'''
	def occursp(u, v):
		pass

	def sigma(u, v, s):
		pass

	def try_subst(u, v, s, ks, kf):
		pass

	def uni(u, v, s, ks, kf):
		pass

	pass

if __name__ == '__main__':
	print("logic module load")


