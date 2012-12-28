#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
import functools

def simpson(fn, upper, lower, divnum=1000):
	'''
	Definite integral by simpson method.
	'''
	h = upper - lower
	if(divnum>0):
		h /= divnum
	
	x = lower
	
	res = 0
	for i in range(divnum):
		res += fn(x) + 4.0*fn(x+h/2) + fn(x+h)
		x += h
	
	return res*h/6

def differentiate(fn, x, dx):
	'''
	Calculate value by differentiate to x power function,
	but an error to some extent.
	fn: Differentiate target function.
	x: Differentiate start point.
	dx: Delta quantity from x.
	'''
	return (fn(x+dx)-fn(x))/dx

def decorator_diffn(fn):
	def _diffn(fn):
		@functools.wraps(fn)
		def __diffn(*args, **kw):
			return differentiate(fn, *args, **kw)
		return __diffn
	
	return _diffn

def diffn(fn):
	'''
	Make function to differentiate given function.
	fn: Differentiate target function.
	'''
	return lambda prob, dx: differentiate(fn, prob, dx)
	
def trapezoid(upper, lower, divnum, func):
	'''
	Integral caluclation by trapezoid approximation.
	'''
	deltax = (lower-upper)/divnum
	x = upper
	s = 0.0
	
	for i in range(divnum-1):
		x = x+deltax
		y = func(x)
		s += y
	
	s = deltax * ((func(upper)+func(lower))/2.0+s)
	
	return s
	
def gamma(z):
	'''
	Gamma function.
	Fractorial generalization.
	Value is approximated by Stirling's approximation.
	'''
	#TODO: improvement precision
	pi = math.pi
	e = math.exp(1)
	
	a = math.sqrt(2*pi/z)
	b = 1/e
	c = 12*z - 1/(10*z)
	
	return a*(b*(z+(1/c)))**z
	
#	def func(t):
#		return math.exp(-t)*t**(z-1)
	
#	return simpson(func, 1, 0)

#Entry point
if __name__ == '__main__':
	print("calculus module load")

