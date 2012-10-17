#!/usr/bin/python3
# -*- coding: utf-8 -*-

import random
import unittest
import math

import pheasant.util as ut

class TestFlatten(unittest.TestCase):
	sample_flatten = [[1,2,3],[4,5,6],["a",[400,500,600],"b"],[7,8,9]]
	fx = [1,2,3,4,5,6,"a",400,500,600,"b",7,8,9]
	
	def test_flatten(self):
		'''test nested list flatten'''
		res = ut.flatten(self.sample_flatten)
		self.assertEqual(res, self.fx)

class TestRands(unittest.TestCase):
	rsize = 100
	
	def test_rands(self):
		'''test get random values'''
		res = ut.rands(501,979,self.rsize)
		self.assertEqual(len(res), self.rsize)

class TestMakeCollection(unittest.TestCase):
	'''
	This is utility of making collection.
	'''
	def test_makeformatlist(self):
		'''
		Test make format list.
		'''
		res = ut.makeformatlist(3, 0)
		chk = [0,0,0]
		self.assertEqual(res, chk)
	
	def test_makelist(self):
		'''
		Test make array 1 dimention.
		'''
		res1 = ut.makelist(10)
		self.assertEqual(len(res1), 10)
		res2 = ut.makelist(100,0)
		self.assertEqual(len(res2), 100)
		
	def test_zipter(self):
		'''
		Test zip and filter function.
		'''
		ls1 = [1,2,3,4]
		ls2 = [1,2,3,4]
		
		res = ut.zipter(ls1, ls2)
		chk = [(1,1),(2,2),(3,3),(4,4)]
		self.assertEqual(chk, res)
		
		res2 = ut.zipter(ls1, ls2, lambda x: x % 2 != 0)
		chk2 = [(1,1),(3,3)]
		self.assertEqual(chk2, res2)

class TestCompose(unittest.TestCase):
	'''
	Function compose test class.
	'''
	def test_compose(self):
		'''function compose test'''
		def square(x):
			return x**2
		
		def double(x):
			return x*2
			
		resfn = ut.compose(square, double)
		res = resfn(2)
		self.assertEqual(res, 8)
		
class TestNearValue(unittest.TestCase):
	'''
	Near value return function test class.
	'''
	def test_nearchoice(self):
		'''test near value return'''
		target = 1
		sample = {
			3 : "Usao",
			5 : "Kitezo",
			7 : "Monchi",
			2 : "Hairegu",
			9 : "Pinny",
			4 : "Goro"
		}
		res = ut.nearchoice(target, sample)
		self.assertEqual(res, "Hairegu")
		
	def test_digitnum(self):
		'''
		Test get number of digits function.
		'''
		self.assertEqual(5, ut.numdigits(12345))
		self.assertEqual(1, ut.numdigits(0))
		self.assertEqual(1, ut.numdigits(-1))
		self.assertEqual(1, ut.numdigits(+1))

class TestBenchMark(unittest.TestCase):
	'''
	Test class for benchmark.
	'''
	def test_tarai(self):
		'''
		Test function for tarai.
		'''
		res1 = ut.tarai(10, 5, 0)
		self.assertEqual(res1, 10)
		#too late.
		#res2 = ut.tarai(18, 12, 6)
		#self.assertEqual(res2, 18)

class TestArrayUtil(unittest.TestCase):
	'''
	Array utility function test class.
	'''
	def test_makeArray(self):
		'''
		Make array.
		Can make multi dementional array too.
		'''
		res = ut.makeArray(2, 2)
		chk = [[None,None],[None,None]]
		self.assertEqual(chk, res)

		res2 = ut.makeArray(3, 4, -1)
		chk2 = [[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]]
		self.assertEqual(chk2, res2)

		res3 = ut.makeArray(4, 3, -1)
		chk3 = [[-1,-1,-1,-1],[-1,-1,-1,-1],[-1,-1,-1,-1]]
		self.assertEqual(chk3, res3)

		class Hoge():
			foo = 0
			
			def __eq__(self, other):
				return self.foo == other.foo
		
		hoge = Hoge()
			
		res4 = ut.makeArray(width=2, height=4, initValue=hoge)
		chk4 = [[hoge,hoge],[hoge,hoge],[hoge,hoge],[hoge,hoge]]
		self.assertEqual(chk4, res4)
		
		self.assertRaises(ValueError, ut.makeArray, 0, 0)
		self.assertRaises(ValueError, ut.makeArray, 1, 0)
		self.assertRaises(ValueError, ut.makeArray, 0, 1)

#Entry point
if __name__ == '__main__':
	print(__file__)
	unittest.main()

