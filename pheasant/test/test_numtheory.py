#!/usr/bin/python3
# -*- coding: utf-8 -*-

import unittest
import math

import pheasant.numtheory as nt

class TestPrimeNumberFunctions(unittest.TestCase):
	'''
	Test class of prime number functions.
	'''
	def test_gcd(self):
		'''
		Test of greatest common divisor function.
		'''
		res = nt.gcd(10, 8)
		
		self.assertEqual(res, 2)
	
	def test_lcm(self):
		'''
		Test of least common multiple function.
		'''
		res = nt.lcm(10, 5)
		self.assertEqual(res, 10)
		res1 = nt.lcm(5, 10)
		self.assertEqual(res1, 10)
		res2 = nt.lcm(12, 18)
		self.assertEqual(res2, 36)
		
	def test_reduct(self):
		'''
		Test of reduction by greatest common divisor.
		'''
		res = nt.reduct(10, 5)
		
		self.assertEqual(res, (2, 1))
		
	def test_coprimep(self):
		'''
		Test coprime function.
		'''
		res = nt.coprimep(2,5)
		self.assertEqual(res, True)
		res = nt.coprimep(2,10)
		self.assertEqual(res, False)
	
	def test_prime(self):
		'''
		Test for prime number check function.
		'''
		res = nt.prime(11)
		self.assertEqual(res, [2,3,5,7,11])
		#res = al.prime(1000000)
		#self.assertEqual(len(res), 78498)
			
	def test_euler_totient(self):
		'''
		Test Euler's totient function.
		'''
		res = nt.euler_totient(100)
		self.assertEqual(res, 40)

	def test_indeq(self):
		'''
		Test of indeterminate equation function.
		'''
		terms = (1004,1001)
		
		res = nt.indeq(terms)
		self.assertEqual(res, (334,-335))		
		
	def test_primedecomp(self):
		'''
		Test prime factor decompose function.
		'''
		res = nt.primedecomp(10)
		chk = {
			2 : 1,
			3 : 0,
			5 : 1,
			7 : 0
		}
		
		self.assertEqual(res, chk)		
		
		n = 25574
		res1 = nt.primedecomp(n)
		chk1 = {}
		ps = nt.prime(n)
		for p in ps:
			chk1[p] = 0
		chk1[2] = 1
		chk1[19] = 1
		chk1[673] = 1

		self.assertEqual(res1, chk1)		

		#Pass test but too late.
		'''
		n2 = 120240
		res2 = nt.primedecomp(n2)
		chk2 = {}
		ps2 = nt.prime(n2)
		for p in ps2:
			chk2[p] = 0
		chk2[2] = 4
		chk2[3] = 2
		chk2[5] = 1
		chk2[167] = 1
		
		self.assertEqual(res2, chk2)		
		'''

		n3 = 256
		res3 = nt.primedecomp(n3)
		chk3 = {}
		ps3 = nt.prime(n3)
		for p in ps3:
			chk3[p] = 0
		chk3[2] = 8
		
		self.assertEqual(res3, chk3)
	
	def test_congruencep(self):
		'''
		Test numbers congruence.
		'''
		res = nt.congruencep(32, 2, 2)
		self.assertEqual(res, True)
		res1 = nt.congruencep(21, 6, 15)
		self.assertEqual(res1, True)
		res1 = nt.congruencep(21, 6, 6)
		self.assertEqual(res1, False)
	
	def test_primep_fermat(self):
		'''
		Test prime number check function.
		Use fermat method.
		'''
		res = nt.primep(2)
		self.assertEqual(res, True)
		res = nt.primep(3)
		self.assertEqual(res, True)
		res = nt.primep(-1)
		self.assertEqual(res, False)
		res = nt.primep(0)
		self.assertEqual(res, False)
		res = nt.primep(1)
		self.assertEqual(res, False)
		res = nt.primep(51)
		self.assertEqual(res, False)
		res = nt.primep(17)
		self.assertEqual(res, True)
		res = nt.primep(31)
		self.assertEqual(res, True)
		res = nt.primep(101)
		self.assertEqual(res, True)
		res = nt.primep(561) #pseudo prime number
		self.assertEqual(res, True)
		
	def test_primep_rabin(self):
		'''
		Test prime number check function.
		This method use Rabin-Miller primality test.
		'''
		mode = "rabin"
		res = nt.primep(2, mode)
		#implement now.
		#self.assertEqual(res, True)
		
class TestIterationMethodTest(unittest.TestCase):
	'''
	Iteration method test class.
	'''
	def test_newton_raphson_by_sqrt(self):
		'''
		Test for newton_raphson method.
		Test by square root.
		'''
		res = nt.newton_raphson(root=2, start=2, repeat=4)
		self.assertEqual(round(res, 7), 1.4142136)

class TestQuadEq(unittest.TestCase):
	'''
	This is quadratic equation test class.
	'''
	def test_quadeq_real_root(self):
		'''
		test a quadratic equation function
		result is real root
		'''
		formula = [1,-2,-3]
		res = nt.quadeq(formula)
		chk = {3,-1}
		
		self.assertEqual(res, chk)

	def test_quadeq_imaginary_root(self):
		'''
		test a quadratic equation function
		result is imaginary root
		but unsupport and should return error.
		'''
		formula = [1,1,1]
		
		self.assertRaises(ValueError, nt.quadeq, formula)
		
	def test_discriminant(self):
		'''
		test for discriminant of quadratic equation function
		'''
		formula1 = [1,-2,-3]
		xnum = nt.discriminant(formula1)
		res1 = xnum > 0
		formula2 = [1,1,1]
		ynum = nt.discriminant(formula2)
		res2 = ynum < 0
		formula3 = [1,-4,4]
		znum = nt.discriminant(formula3)
		res3 = znum == 0
		self.assertEqual(res1, True)
		self.assertEqual(res2, True)
		self.assertEqual(res3, True)
		
class TestSleq(unittest.TestCase):
	formura1 = [[2,-3,-13],[7,5,1]]
	equation1 = [-2,3]

	formura2 = [[1,-1,2,0],[-2,3,-5,1],[1,-1,1,0]]
	equation2 = [1,1,0]
	
	formura2_2 = [[1,5,-4,-1],[1,-5,15,17],[4,9,5,16]]
	equation2_2 = [-3,2,2]
	
	formura3 = [[1,-2,3],[3,-6,9]]
	
	formura4 = [[1,-2,3],[3,-6,10]]
	
	formura5 = [[1,2,-5,4],[2,3,-7,7],[4,-1,7,7]]
	
	formura6 = [[1,2,-5,4],[2,3,-7,7],[4,-1,7,8]]
	
	INDEFINITE = "indefinite"
	IMPOSSIBLE = "impossible"
	
	def test_sleq_eq_2dim(self):
		'''test by 2 dimention'''
		result = nt.sleq(self.formura1)
		self.assertEqual(result, self.equation1)

	def test_sleq_eq_3dim_inverse_matrix_error(self):
		'''test by 3 dimention at inverse matrix error occured.'''
		result = nt.sleq(self.formura2)
		res = map(lambda x: round(x), result)
		self.assertEqual(list(res), self.equation2)
	
	def test_sleq_eq_3dim(self):	
		'''test by 3 dimention'''
		result2 = nt.sleq(self.formura2_2)
		res2 = map(lambda x: round(x), result2)
		self.assertEqual(list(res2), self.equation2_2)

	def check_errormessage(self, func, args, chkmessage):
		'''check returned error message'''
		try:
			func(args)
		except ValueError as ex:
			message = str(ex)
			self.assertEqual(message, chkmessage)

	def test_sleq_eq_2dim_indefinite(self):
		'''test by 2 dimention and indefinite case'''
		self.check_errormessage(nt.sleq, self.formura3, self.INDEFINITE)

	def test_sleq_eq_2dim_inpossible(self):
		'''test by 2 dimantion and impossible case'''
		self.check_errormessage(nt.sleq, self.formura4, self.IMPOSSIBLE)

	def test_sleq_eq_3dim_indefinite(self):
		'''test by 3 dimention and imdefinite case'''
		self.check_errormessage(nt.sleq, self.formura5, self.INDEFINITE)

	def test_sleq_eq_3dim_inpossible(self):
		'''test by 3 dimantion and inpossible case'''
		self.check_errormessage(nt.sleq, self.formura6, self.IMPOSSIBLE)

#	def test_sleq_indefinite_return_formula(self):
#		'''test return formula at indefinite case'''
#		forms = [[1,2,4,0],[2,-3,1,0],[0,1,1,0]]
#		
#		resforms = ut.sleq(forms)
#		
#		chkforms = [[1,0,2,0],[0,1,1,0],[0,0,0,0]]
#		self.assertEqual(resforms, chkforms)

class TestFormula(unittest.TestCase):
	'''
	Test class for formula class.
	'''
	def test_term_add(self):
		'''
		Term class add test.
		'''
		a = nt.Term(2,2)	#2**2
		b = nt.Term(3,2)	#3**2
		
		res = a+b
		
		self.assertEqual(res.calc(), 13)

class TestProgression(unittest.TestCase):
	'''
	Progression test class.
	'''
	def test_zeta(self):
		'''
		Zeta function test.
		'''
		#res = nt.zeta(2)		
		#self.assertEqual(res, math.pi**2/6)
		#TODO Implement method thinking now.
		pass
		
class TestEvenOdd(unittest.TestCase):
	'''
	Test class to judge even or odd functions.
	'''
	def test_even(self):
		'''
		Even number test.
		'''
		r1 = nt.even(2)
		self.assertEqual(r1, True)
		r2 = nt.even(0)
		self.assertEqual(r2, True)
		r3 = nt.even(7)
		self.assertEqual(r3, False)
		
	def test_odd(self):
		'''
		Odd number test.
		'''
		r1 = nt.odd(3)
		self.assertEqual(r1, True)
		r3 = nt.odd(0)
		self.assertEqual(r3, False)
		
class TestCollatz(unittest.TestCase):
	'''
	Collatz probrem test class.
	'''
	def test_collatz(self):
		'''
		Collatz probrem function test.
		'''
		one = 1
		
		r1 = nt.collatz(10)
		self.assertEqual(r1, one)
		r2 = nt.collatz(31)
		self.assertEqual(r2, one)
		r3 = nt.collatz(3*2**53)
		self.assertEqual(r3, one)
		
class TestHarmonicSeries(unittest.TestCase):
	'''
	Harmonic series test class.
	'''
	def test_harmony(self):
		'''
		Harmonic series function test.
		'''
		har = nt.harmony()
		
		self.assertEqual(next(har), 1/1)
		self.assertEqual(next(har), 1/2)
		self.assertEqual(next(har), 1/3)

		har = nt.harmony(101)
		
		self.assertEqual(next(har), 1/101)
		self.assertEqual(next(har), 1/102)
		self.assertEqual(next(har), 1/103)

class TestNumericalSequence(unittest.TestCase):
	'''
	Test class for numerical sequence functions.
	'''
	def test_addchain(self):
		'''
		Test addition chain function.
		'''
		res = nt.addchain(45)
		chk = [1,2,4,5,10,11,22,44,45]
		
		self.assertEqual(res, chk)
		
	def test_expop(self):
		'''
		Test expop function.
		'''
		res = nt.expop(4)
		self.assertEqual(res, True)
		res = nt.expop(100)
		self.assertEqual(res, True)
		res = nt.expop(101)
		self.assertEqual(res, False)
		res = nt.expop(998001)
		self.assertEqual(res, True)
		res = nt.expop(0)
		self.assertEqual(res, False)
		res = nt.expop(-1)
		self.assertEqual(res, False)

class TestInversion(unittest.TestCase):
	'''
	Test inversion number function.
	'''
	def test_inversion(self):
		res = nt.inversion(3421)		
		self.assertEqual(5, res)

		res2 = nt.inversion(365142)		
		self.assertEqual(10, res2)

if __name__ == '__main__':
	print(__file__)
	unittest.main()

