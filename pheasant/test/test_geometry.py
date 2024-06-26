#!/usr/bin/python3
# -*- coding: utf-8 -*-

import unittest
import math

import pheasant.geometry as gm
import pytest

class TestCalcArea(unittest.TestCase):
	'''
	Test class for calculate area functions.
	'''
	def test_heron(self):
		'''
		Test heron function.
		'''
		res = gm.heron(2,2,2)
		self.assertEqual(res, 2*math.sqrt(3)/2)
		
	def test_trianglep(self):
		'''
		Test to check Triangle fomation condition satisfied.
		'''
		tri1 = gm.trianglep(2,2,2)
		self.assertEqual(tri1, True)
		tri2 = gm.trianglep(2,2,100)
		self.assertEqual(tri2, False)
		tri3 = gm.trianglep(2,100,2)
		self.assertEqual(tri3, False)
		tri4 = gm.trianglep(100,2,2)
		self.assertEqual(tri4, False)
		
class TestCoodinateTransform(unittest.TestCase):
	'''
	Test class for coordinate transform.
	'''
	def test_polar2rectangular(self):
		'''
		Test function for converting polar to rectangular.
		'''
		res = gm.polar2rectangular(1.0, 0)
		self.assertEqual(res, (1.0, 0.0))
		res = gm.polar2rectangular(0.0, 1.0)
		self.assertEqual(res, (0.0, 0.0))
		#Lower test exists an error.
		#res = gm.polar2rectangular(1.0, 1.57)
		#self.assertEqual(res, (7.9639349e-04, 0.9999997))
		#res = gm.polar2rectangular(4.0, 3.14159)
		#self.assertEqual(res, (-4.000000, 1.1094401e-05))

class TestPythagoreanTheorem(unittest.TestCase):
	'''
	Pythagorean theorem function test class.
	'''
	def test_pythagonum(self):
		'''
		Test pythagonum function.
		'''
		res = gm.pythagonum(2,1)
		self.assertEqual(res, (3,4,5))
		res = gm.pythagonum(17,10)
		self.assertEqual(res, (189,340,389))
		res = gm.pythagonum(26,23)
		self.assertEqual(res, (147,1196,1205))
		res = gm.pythagonum(4,2)
		self.assertEqual(res, ())

class TestCoordinateTransformation(unittest.TestCase):
	'''
	Coordinate transfomation test class.
	'''
	def test_geo2rect(self):
		'''
		Test convertion function for geographic coodinates to rectangular coodinates.
		'''
		pass

def test_distance():
	p1 = gm.Point(1, 1)
	p2 = gm.Point(4, 5)
	result = gm.get_distance(p1, p2)
	assert result == 5

def get_sample_line():
	p1 = gm.Point(1, 3)
	p2 = gm.Point(4, 9)
	return gm.Line([p1, p2])

def test_get_slope():
	line = get_sample_line()
	slope = gm.get_slope(line)
	assert slope == 2

def test_get_intercept():
	line = get_sample_line()
	intercept = gm.get_intercept(line)
	assert intercept == 1

def test_line_crosspoint():
	line1 = gm.Line([gm.Point(-1, -1), gm.Point(1, 1)])
	line2 = gm.Line([gm.Point(1, -1), gm.Point(-1, 1)])
	result = gm.get_line_crosspoint(line1, line2)
	assert result == gm.Point(0, 0)

@pytest.mark.skip(reason='ウインドウが自動で閉じられないためスキップ')
def test_plot_graph():
	yokohama_bbox = [35.46537173052059, 35.44347093055431,
					139.64015389192105, 139.62228254640752]
	gm.plot_drive_graph(bbox=yokohama_bbox)	

if __name__ == '__main__':
	print(__file__)
	unittest.main()
