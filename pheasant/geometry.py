#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math

import pheasant.numtheory as nt

def heron(a, b, c):
	'''
	Calculate area by heron method.
	'''
	s = (a+b+c)/2
	
	return math.sqrt(s*(s-a)*(s-b)*(s-c))
	
def trianglep(a, b, c):
	'''
	Check triangle formation condition satisfied.
	'''
	return a<b+c and b<a+c and c<a+b

def polar2rectangular(r, rad):
	'''
	Polar to rectangular coordinate transform.
	r: polar coordinates value
	rad: radian value
	'''
	return (r*math.cos(rad), r*math.sin(rad))

def pythagonum(m, n):
	'''
	Calculate pythagorean number.
	'''
	if nt.coprimep(m, n) and nt.odd(m-n):
		return (m**2-n**2, 2*m*n, m**2+n**2)
	else:
		return ()

class CoordinateTransformer():
	'''
	Coordinate transformation class.
	'''
	@classmethod
	def geo2rect(cls, lon, lat, elheight):
		'''
		Convert from geographic coodinates to rectangular coodinates.
		lon: Longitude
		lat: Latitude
		elheight: Ellipsoidal height
		'''
		pass

class Point():
	def __init__(self, x=0, y=0):
		self.x = x
		self.y = y

def get_distance(p1: Point, p2: Point):
	'''
	p1とp2の距離を計算します。
	'''
	return ((p2.x - p1.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

class Line():
	def __init__(self, points: [Point]):
		self.points = points
		self.first = points[0]
		self.last = points[len(points) - 1]

def get_slope(line: Line):
	first = line.first
	last = line.last
	slope = (last.y - first.y) / (last.x - first.x)
	return slope

def get_intercept(line: Line, slope=None):
	if (slope == None):
		slope = get_slope(line)
	first = line.first
	intercept = first.y - slope * first.x
	return intercept

def get_line_closspoint(line1: Line, line2: Line) -> Point:
	slope1 = get_slope(line1)
	intercept1 = get_intercept(line1, slope1)
	slope2 = get_slope(line2)
	intercept2 = get_intercept(line2, slope2)
	x = (intercept2 - intercept1) / (slope1 - slope2)
	y = slope1 * x + intercept1
	return Point(x, y)	

if __name__ == '__main__':
	print("geometry module load")

