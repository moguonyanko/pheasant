#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math

class Element():
	'''
	Element in group.
	'''
	def operate():
		'''
		Carry out an operation.
		'''
		pass

class Axiom():
	'''
	An axiom, group, ring, field...
	'''
	def satisfy(target):
		'''
		Is target satisfy the axiom?
		'''
		pass
	
	def define():
		'''
		Get initializer of algebraic structure correspond to the axiom. 
		'''
		pass

class GroupAxiom(Axiom):
	'''
	An group axiom.
	'''
	def satisfy(target):
		'''
		Is target satisfy group axiom?
		'''
		pass
		
	def define():
		'''
		Get definition of algebraic structure, in a word Group class return.
		'''
		return Group

class AlgebraicStructureFactory():
	'''
	Abstract Factory to make algebraic structure.
	'''
	def create(axiom, elements):
		'''
		If elements satisfy axiom, make structure according to the axiom.
		'''
		pass

class Group():
	'''
	Group expression class.
	'''
	def __init__(elements):
		'''
		Element list given.
		'''
		self.elements = elements	

class Ring():
	'''
	Ring expression class.
	'''
	pass

class Field():
	'''
	Field expression class.
	'''
	pass


