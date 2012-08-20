#!/usr/bin/python3
# -*- coding: utf-8 -*-

import unittest

import pheasant.logic as lo

class TestUnify(unittest.TestCase):
	'''
	Unify test class.
	Example, pattern matching utillity etc.
	'''
	def test_symbolsetitem(self):
		'''
		Symbol setitem test.
		Should raise error.
		'''
		sym = lo.Symbol("HOGE")
		
		self.assertRaises(AttributeError, setattr, sym, "value", "FOO")
		self.assertEqual(sym.value, "HOGE")
		
	def test_unify(self):
		'''
		Unify function test.
		'''
		pass

class TestNatualNumber(unittest.TestCase):
	'''
	Definition of Natural Number test class.
	'''
	def test_natp(self):
		'''
		Natural Number predicate test.
		'''
		res = lo.natp(0)
		self.assertEqual(res, True)
		res = lo.natp(-1)
		self.assertEqual(res, False)
		res = lo.natp(100)
		self.assertEqual(res, True)
		res = lo.natp(100.001)
		self.assertEqual(res, False)
		#Maximum recursion depth exceeded in comparison
		#res = lo.natp(999999)
		#self.assertEqual(res, True)

if __name__ == '__main__':
	print(__file__)
	unittest.main()		

