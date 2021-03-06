#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numbers
import math
import copy

import pheasant.util as ut
import pheasant.numtheory as nt

class Vector():
	'''
	Vector class definition.
	This class will be used Matrix.
	'''
	
	'''
	Column values list as vector elements or matrix columns.
	'''
	cols = []
	
	'''
	Significant figure.
	'''
	SIGNIFICANT_FIGURE = 4
	
	def __init__(self, cols):
		'''
		Initialize vector.
		cols: column value list.
		'''
		self.cols = cols
		
	def __len__(self):
		'''
		Vector elements length return.
		'''
		return len(self.cols)
		
	def __getitem__(self, index):
		'''
		Get vector component by numeric index.
		'''
		return self.cols[index]

	def __setitem__(self, index, value):
		'''
		Set vector component by numeric index.
		'''
		self.cols[index] = value
		
	def __common_mul_operate(self, target, org):
		'''
		Common multiplication operation.
		If argument is numbers.Real, scalar multiplication.
		Elif Vector, dot multiplication.
		'''
		if isinstance(target, numbers.Real):
			ps = [target*a for a in org.cols]
			return Vector(ps)
		elif isinstance(target, Vector):
			tmp = zip(org.cols, target.cols)
			res = 0
			for a1, a2 in tmp:
				res += a1*a2
			return res
		else:
			raise ValueError("Argument need to be numbers.Real or Vector.")
	
	def __add__(self, target):
		'''
		Addition vector.
		'''
		cmps = zip(self.cols, target.cols)
		ps = [a+b for a,b in cmps]
		
		return Vector(ps)
		
	def __sub__(self, target):
		'''
		Subtract vector.
		'''
		cmps = zip(self.cols, target.cols)
		ps = [a-b for a,b in cmps]
		
		return Vector(ps)
		
	def __mul__(self, target):
		'''
		Multiplication vector for left side vector.
		'''
		return self.__common_mul_operate(target, self)

	def __rmul__(self, target):
		'''
		Multiplication vector for right side vector.
		'''
		return self.__common_mul_operate(target, self)
		
	def __eq__(self, target):
		'''
		If point value equal, vectors are equal.
		'''
		if target == None: return False
		
		size = len(self.cols)
		targetsize = len(target.cols)
		if size > 0 and targetsize > 0 and size == targetsize:
			cmps = zip(self.cols, target.cols)
			sg = self.SIGNIFICANT_FIGURE
			for a,b in cmps:
				if round(a, sg) != round(b, sg):
					return False
			return True
		else:
			return False
		
	def __str__(self):
		'''
		Return point value by string expression.
		'''
		return str(self.cols)

	def __sizecheck(self):
		'''
		Dimention size check.
		'''
		if len(self.cols) != 2:
			raise ValueError("Sorry, now required 2 dimention vector.")

	def rotate(self, degree):
		'''
		Calculate rotate matorix or vector expression.
		note:This function is adapt only 2 dimention vector.
		'''
		self.__sizecheck()
		
		rad = math.radians(degree)
		v1 = Vector([math.cos(rad), math.sin(rad)])
		v2 = Vector([-math.sin(rad), math.cos(rad)])
		rotatem = Matrix([v1, v2])
		
		return rotatem*self

	def turn(self, degree):
		'''
		Turn vector function.
		note:This function is adapt only 2 dimention vector.
		'''
		self.__sizecheck()
			
		rad = math.radians(2*degree)
		v1 = Vector([math.cos(rad), math.sin(rad)])
		v2 = Vector([math.sin(rad), -math.cos(rad)])
		rotatem = Matrix([v1, v2])
		
		return rotatem*self
		
	def normalize(self):
		'''
		Vector normalization.
		'''
		orgcols = self.cols
		tmp = [a**2 for a in orgcols]
		distance = math.sqrt(sum(tmp))
		newcols = list(map(lambda x: x*1/distance, orgcols))
		
		return Vector(newcols)
		
	def orthproj(self, targetv, normal=False):
		'''
		Orthographic projection vector.
		'''
		normv = self
		if normal == False: 
			normv = self.normalize()
			
		return targetv*normv*normv

def normalize(vec):
	'''
	Vector normalization.
	'''
	return vec.normalize()

def orthproj(scalev, targetv):
	'''
	Orthographic projection vector.
	'''
	return scalev.orthproj(targetv)

def rmcols(targetv, vecs):
	'''
	Remove columns from vector.
	'''
	orgv = Vector(targetv.cols)
	for v in vecs:
		orgv = orgv-v
		
	return orgv

def orthogonalize(vecs):
	'''
	Gram-Schmidt orthogonalization.
	'''
	normvs = [vecs[0].normalize()]
	tmp = vecs[1:len(vecs)]
	for v in tmp:
		orthvs = []
		for normv in normvs:
			otv = normv.orthproj(v, normal=True)
			orthvs.append(otv)
		rmdv = rmcols(v, orthvs)
		nmv = rmdv.normalize()
		normvs.append(nmv)
		
	return normvs

class Matrix():
	'''
	Matrix class definition.
	This is used to express linear mapping.
	'''
	
	'''
	Vector list as matrix rows.
	'''
	rows = []

	'''
	Exception messages.
	'''
	invalid_index_message = "Invarid index recieved."
	
	def __init__(self, rows):
		'''
		Initialize matrix by received row vector.
		'''
		self.rows = rows
		
	def __setitem__(self, row_column, value):
		'''
		Set value to matrix by row column tuple.
		If row_column is tuple, set element. 
		Tuple is composed (column index, row index).
		If row_column is int, set vector. 
		'''
		if isinstance(row_column, tuple):
			self.rows[row_column[1]][row_column[0]] = value
		elif isinstance(row_column, int):
			self.rows[row_column] = value
		else:
			raise ValueError(self.invalid_index_message)
		
	def __getitem__(self, row_column):
		'''
		Get value from matrix by row column tuple.
		If row_column is tuple, return element. 
		Tuple is composed (column index, row index).
		If row_column is int, return row of vector. 
		'''
		if isinstance(row_column, tuple):
			return self.rows[row_column[1]][row_column[0]]
		elif isinstance(row_column, int): #Get row.
			return self.rows[row_column]
		else:
			raise ValueError(self.invalid_index_message)
		
	def __add__(self, target):
		'''
		Matrix addition.
		'''
		newrows = []
		rows = self.rows
		trows = target.rows
		rowlen = len(rows)
		
		for i in range(rowlen):
			newrow = rows[i]+trows[i]
			newrows.append(newrow)
			
		return Matrix(newrows)
		
	def __sub__(self, target):
		'''
		Matrix subsctiption.
		'''
		newrows = []
		rows = self.rows
		trows = target.rows
		rowlen = len(rows)
		
		for i in range(rowlen):
			newrow = rows[i]-trows[i]
			newrows.append(newrow)
			
		return Matrix(newrows)

	def __innermul(self, col, nums):
		'''
		Matrix column multipulate to numbers.
		'''
		res = []
		for i in range(len(nums)):
			res.append(col[i]*nums[i])
		return sum(res)
	
	def __mulvector(self, target):
		'''
		Matrix multipulate vector.
		'''
		orgcols = [v.cols for v in self.rows]
		cols = list(zip(*orgcols))
		newcols = list(map(lambda col: self.__innermul(col, target.cols), cols))
		return Vector(newcols)

	def __common_mulmatscala(self, target):
		'''
		Matrix multipulate scala common routine.
		'''
		vecs = []
		for v in self.rows:
			vecs.append(Vector([x*target for x in v.cols]))
		return Matrix(vecs)

	def __mul__(self, target):
		'''
		Composite linear mapping.
		'''
		if isinstance(target, Vector):
			if len(self.rows) != len(target.cols):
				raise ValueError("difference size of matrix rows and vector columns.")
			return self.__mulvector(target)
		elif isinstance(target, Matrix):
			if len(self.rows) != len(target.rows[0].cols):
				raise ValueError("difference size of matrix rows and matrix columns.")
			return Matrix([self.__mulvector(trow) for trow in target.rows])
		elif isinstance(target, numbers.Real):
			return self.__common_mulmatscala(target)
		else:
			pass
			
	def __rmul__(self, target):
		'''
		Composite linear mapping by right multipulate.
		'''
		if isinstance(target, numbers.Real):
			return self.__common_mulmatscala(target)
		else:
			pass
			
	def __propereq_dim2(self, n):
		'''
		Solve to power of 2x2 dimention matrix by proper equation.
		n: Power number.
		'''
		eg = self.eigen()
		egvalues = sorted(list(eg.keys()))
		
		a = egvalues[0]
		b = egvalues[1] 
		E = einheit(2)
		
		resm = ((b**n-a**n)/(b-a))*self+((a**n*b-a*b**n)/(b-a))*E
		
		return resm
		
	def __pow__(self, target):
		'''
		If power number is -1, inverse matrix.
		Dimention 2 or more than 3, method change.
		target: Power number.
		'''
		is2x2 = self.dim() == [2,2]
		if target != -1:
			if is2x2:	return self.__propereq_dim2(target)
			else:	raise ValueError("Unsupported exponent value.")
		else:
			cols = self.rows[0].cols
			if is2x2:
				detval = det(self)
				if detval == 0:	raise ValueError("no matrix determinent.")
				
				v1 = self.rows[0]
				v2 = self.rows[1]
				a = v1.cols[0]
				b = v1.cols[1]
				c = v2.cols[0]
				d = v2.cols[1]
				scala = 1/detval
				newv1 = Vector([d,-b])
				newv2 = Vector([-c,a])
			
				return scala*Matrix([newv1,newv2])
			elif self.squarep():
				E = einheit(len(self.rows))
				eqvecs = []
				veccols = [row.cols for row in self.rows]
				for erow in E.rows:
					tmp = veccols+[erow.cols]
					formuras = list(zip(*tmp))
					eqs = ut.sleq(formuras)
					eqvecs.append(Vector(eqs))
			
				return Matrix(eqvecs)
			else:
				pass			
	
	def __eq__(self, target):
		'''
		If rows equal, return true.
		'''
		if target == None: return False
		
		rows = self.rows
		tarrows = target.rows
		rowlen = len(rows)

		if rowlen != len(tarrows):
			return False
			
		for i in range(rowlen):
			if rows[i] != tarrows[i]:
				return False
				
		return True
				
	def __str__(self):
		'''
		Rows string expression return.
		'''
		rows = self.rows
		strs = ""
		for row in rows:
			strs += str(row)
		return strs
		
	def __round__(self, n=0):
		'''
		Matrix element round.
		'''
		#TODO: not make new Matrix.
		newrows = []	
		for row in self.rows:
			newrows.append(Vector([round(value, n) for value in row.cols]))
			
		return Matrix(newrows)
		
	def __len__(self):
		'''
		Matrix row length return.
		'''
		rowslen = len(self.rows)
		
		colslen = 0
		try:
			v1 = self[0]
			if v1 != None: 
				colslen = len(v1)
		except IndexError as ex:
			return 0
		
		return min(rowslen, colslen)
		
	def find(self, pred):
		'''
		Find target element to fulfill the given predicate.
		'''	
		row = 0
		col = 0
		rsize = len(self.rows)
		csize = len(self[row])
		
		retcol = 0
		retrow = 0
		value = self[(col,row)]
		while row < rsize:
			col = 0
			maxcol = 0
			while col < csize:
				if pred(self[(col,row)],value):
					retcol = col
					retrow = row
					value = self[(col,row)]
				col += 1
			row += 1
		
		return (retcol, retrow, value)		
	
	def minElement(self):
		'''
		Get minimum element in this matrix.
		'''
		def pred(now, target):
			return now < target
		
		return self.find(pred)
		
	def maxElement(self):
		'''
		Get maximum element in this matrix.
		'''
		def pred(now, target):
			return now > target
		
		return self.find(pred)
		
	def rotate(self, degree=0):
		'''
		Expression matrix of rotated degree this matrix.
		'''
		pass
		
	def turn(self, degree):
		'''
		Turn expression matrix.
		'''
		pass
		
	def squarep(self):
		'''
		Predicate to confirm matrix square.
		'''
		rownum = len(self.rows)
		colnum = len(self.rows[0].cols)
		
		return rownum == colnum
		
	def dim(self):
		'''
		Matrix dimension return.
		'''
		return [len(self.rows), len(self.rows[0].cols)]
		
	def symmetryp(self):
		'''
		Predicate to confirm matrix symmetry.
		'''
		if self.squarep() == False:
			return False
			
		for i in range(len(self.rows)):
			for j in range(len(self.rows[i].cols)):
				if i != j and (self.rows[i].cols[j] != self.rows[j].cols[i]):
					return False	
					
		return True
		
	def get_column(self, column_num):
		'''
		Get matrix column.
		'''
		res = []
		
		for row in self:
			for n, col in enumerate(row):
				if n == column_num:
					res.append(col)
		
		return res		
		
	def get_columns(self):
		'''
		Return matrix elements by columns form.
		'''
		rows = [v.cols for v in self]
		
		return [list(r) for r in zip(*rows)]
		
	def swap(self, i, j, target="column"):
		'''
		Swap by column or row index.
		'''
		if i == j: return
		
		if target == "column":
			cols = self.get_columns()
			
			#Swap matrix columns.		
			tmp = cols[i]
			cols[i] = cols[j]
			cols[j] = tmp
			#TODO: Is need Vector create?
			self[i] = Vector(cols[i])
			self[j] = Vector(cols[j])
		else: #Swap matrix rows. 
			tmp = self[i]
			self[i] = self[j]
			self[j] = tmp
	
	def det(self):
		'''
		Caluculate determinant.
		'''
		dimension = self.dim()
		if dimension == [2,2]:
			v1 = self.rows[0]
			v2 = self.rows[1]
			a = v1.cols[0]
			b = v1.cols[1]
			c = v2.cols[0]
			d = v2.cols[1]
		
			return a*d-b*c
		
		elif dimension == [3,3]:
			row1 = self.rows[0]
			row2 = self.rows[1]
			row3 = self.rows[2]
		
			A = row1.cols[0]
			B = row1.cols[1]
			C = row1.cols[2]
			x = row2.cols[0]
			y = row2.cols[1]
			z = row2.cols[2]
			a = row3.cols[0]
			b = row3.cols[1]
			c = row3.cols[2]
		
			return A*y*c+B*z*a+C*x*b-A*z*b-B*x*c-C*y*a
		
		else:#TODO: Not work...
			_mat = self.decomp()
			
			reop = []
			for i, col in enumerate(_mat):
				for j, ele in enumerate(col): 
					if i == j: reop.append(ele)
			
			return ut.mac(reop)			
			
	def __eigen_2dim(self):
		'''
		Calculate eigen value for 2 dimension matrix.
		'''
		formula = ut.makelist(length=3, initvalue=0)
		formula[0] = 1 #x^2 coef

		rows = self.rows
		cols1 = rows[0].cols
		cols2 = rows[1].cols
		a = cols1[0]
		d = cols2[1]
		formula[1] = a*-1+(-1)*d

		formula[2] = self.det()
	
		egs = nt.quadeq(formula)
		
		if len(egs) <= 1:
			raise ValueError("Multiple root cannot deal.")
		
		res = {}
		for eg in egs:
			resA = a-eg
			resC = cols2[0]
	#TODO: reduct function is fault.
	#		if resA != 1 and resC != 1:
	#			if resA>resC:
	#				resA,resC = nt.reduct(resA, resC)
	#			elif resC<resA:	
	#				resC,resA = nt.reduct(resC, resA)
	#			else:
	#				resA = resC = 1
				
			res[eg] = Vector([-resC,resA])
	
		return res
	
	def __eigen_3dim(self):
		'''
		Calculate eigen value for 3 dimension matrix.
		'''
		#TODO:More than 3 dimention case.
		pass			
		
	def eigen(self):
		'''
		Calculate eigen value.
		Return dict has key of eigen value, 
		value of eigen vector.
		'''
		#if self.symmetryp():
		#	return self.__jacobi()
			
		dimension = self.dim()
		if dimension == [2,2]:
			return self.__eigen_2dim()
		elif dimension == [3,3]:
			return self.__eigen_3dim()
		else:
			raise ValueError("Sorry, unsupported dimension.")
		
	def base_exchange(self, dist):
		'''
		Matrix base exchange.
		'''
		return self**-1*dist
	
	def base_exchanged_express(self, exchange_mat):
		'''
		Matrix expression after base exchange.
		From standard base to any base.
		'''
		return exchange_mat**-1*self*exchange_mat
		
	def diagonalize(self):
		'''
		Matrix diagonalization.
		Complex implement for check calculation.
		'''
		eg = self.eigen()
		vecs = [eg[egv] for egv in sorted(eg)]
		if self.symmetryp():
			normvecs = [normalize(vec) for vec in vecs]
			normmat = Matrix(normvecs)
			transnormmat = normmat.transpose()
			return transnormmat*self*normmat
		else:
			egmat = Matrix(vecs)
			return egmat**-1*self*egmat	
		
	def transpose(self):
		'''
		Matrix transposition.
		'''
		cols = [row.cols for row in self.rows]
		tmp = list(zip(*cols))
		newrows = [Vector(newcols) for newcols in tmp]
		
		return Matrix(newrows)
	
	def trace(self):
		'''
		Trace of matrix.
		'''
		egvecs = self.eigen()
		return sum([egvalue for egvalue in egvecs])	
	
	def sum_elements(self):
		'''
		Calculate sum of elements.
		'''
		r = len(self)
		c = len(self[0])
		
		res = []
		for i in range(r):
			for j in range(c):
				res.append(self[(i, j)])
		
		return sum(res)
	
	def decomp(self):
		'''
		Matrix decompose by LU decompose method.
		'''
		return lu_decompose(self)
		
def jacobi(mat):
	'''
	Caluclate matrix eigen value and vector by Jacobi method.
	mat: Symmetry matrix. If that is not symmetry, raise ValueError.
	'''
	#TODO: After implement, this function is concealed.
	if mat.symmetryp() == False:
		raise ValueError("Matrix is not symmetry!")
		
	EPS = 0.0001
	
	matsize = len(mat)
	rng = range(matsize)
	eigmat = einheit(matsize)
	
	maxEle = mat.maxElement()
	
	testcounter = 0 #value check counter
	while maxEle[2] >= EPS:
	
		#if testcounter > 10: break
		#testcounter += 1
		
		p = maxEle[0]
		q = maxEle[1]
		app = mat[(p,p)]
		apq = mat[(p,q)]
		aqq = mat[(q,q)]
		
		alpha = (app - aqq)/2
		beta = -apq
		gamma = abs(alpha)/math.sqrt(alpha**2 + beta**2) #TODO: beta is OverflowError

		sin = math.sqrt((1 - gamma)/2)
		cos = math.sqrt((1 + gamma)/2)
		if alpha*beta < 0: sin = -sin
		
		for i in rng: #row update
			temp = cos*mat[(p,i)] - sin*mat[(q,i)]
			mat[(q,i)] = sin*mat[(p,i)] + cos*mat[(q,i)]
			mat[(p,i)] = temp

		for i in rng: #column update
			mat[(i,p)] = mat[(p,i)]
			mat[(i,q)] = mat[(q,i)]
		
		mat[(p,p)] = cos*cos*app + sin*sin*aqq - 2*sin*cos*apq
		mat[(p,q)] = sin*cos*(app - aqq) + (cos*cos - sin*sin)*apq
		mat[(q,p)] = mat[(p,q)]
		mat[(q,q)] = sin*sin*app + cos*cos*aqq + 2*sin*cos*apq #TODO: Bug in apq?
	
		for i in rng:
			temp = cos*eigmat[(i,p)] - sin*eigmat[(i,q)]
			eigmat[(i,q)] = sin*eigmat[(i,p)] + cos*eigmat[(i,q)]
			eigmat[(i,p)] = temp
			
		maxEle = mat.maxElement()

	eigs = {}
	for i in rng:
		eigs[mat[(i,i)]] = eigmat[i]
	
	return eigs

'''		
def jacobi(mat):
	if mat.symmetryp() == False:
		raise ValueError("Matrix is not symmetry!")
		
	EPS = 0.0001
	
	matsize = len(mat)
	rng = range(matsize)
	eigmat = einheit(matsize)
	
	maxEle = mat.maxElement()
	
	pass
'''
		
def einheit(dim):
	'''make identity matrix'''
	rows = ut.makeformatlist(dim, None)
	for i in range(dim):
		row = ut.makeformatlist(dim, 0)
		row[i] = 1
		rows[i] = Vector(row)
	return Matrix(rows)
	
def det(mat):
	'''
	Matrix determinent.
	todo: more than 3 dimention matrix
	'''
	return mat.det()

def make_empty_matrix(size, init_value=0):
	'''
	Utility function to make empty matrix.
	'''
	vs = []
	for i in range(size):
		vs.append(Vector([init_value]*size))
	
	_mat = Matrix(vs)
	
	return _mat

def lu_decompose(mat):
	'''
	LU-decomposition of matrix.
	'''
	if mat == None: 
		raise ValueError("Matrix is none. ")

	size = len(mat)
	
	if size <= 0:
		raise ValueError("Matrix elements are empty. ")
	
	_mat = copy.deepcopy(mat) #Heavy process but making sure...
	
	for k in range(size):
		#Pivot exchange
		maxcol = k
		max_ele = _mat[(k, k)]
		for m in range(k, size):
			if max_ele < abs(_mat[(k, m)]):
				maxcol = m
				max_ele = _mat[(k, m)]
		_mat.swap(k, maxcol)
		
		#Decompose L
		#'reopposite' is the reciprocal of the opposite angle element.
		#In L decomposition numerator is got at U decomposition once previous.
		reopposite = 1.0 / _mat[(k, k)]
		for i in range(k+1, size):
			_mat[(i, k)] = _mat[(i, k)] * reopposite
			
		#Decompose U
		for i in range(k+1, size): #Loop of column.
			for j in range(k+1, size): #Loop of row.
				#Preserve the opposite angle element of U to original that.
				#This '_mat[(i, j)]' is '_mat[(i, k)]' in next L decomposition.
				_mat[(i, j)] = _mat[(i, j)] - _mat[(i, k)] * _mat[(k, j)]
		
	return _mat
	
def spectral_decompose(mat):
	'''
	Matrix spectral decomposition.
	'''
	diag = mat.diagonalize()
	matdim = mat.dim()
	return [diag[(i,i)] for i in range(matdim[0])]
	
def base_exchange(matSrc, matDist):
	'''
	Matrix base exchange.
	'''
	return matSrc.base_exchange(matDist)
	
def base_exchanged_express(express_mat, exchange_mat):
	'''
	Matrix expression after base exchange.
	From standard base to any base.
	'''
	return express_mat.base_exchanged_express(exchange_mat)

def eigen(mat):
	'''
	Calculate eigen value.
	'''
	return mat.eigen()
		
def diagonalize(mat):
	'''
	Matrix diagonalization.
	Complex implement for check calculation.
	'''
	return mat.diagonalize()
	
def trace(mat):
	'''
	Trace of matrix.
	'''
	return mat.trace()
	
def transpose(mat):
	'''
	Matrix transpose.
	'''
	return mat.transpose()

def sweep_out(leftm, rightv):
	'''
	Solution of simultanious linear equations 
	by sweep out method.
	Take account of pivot.
	'''
	#TODO: implement on the way.
	eps = 1E-8
	msize = len(leftm)
	newvs = leftm.rows + [rightv]
	
	mat = Matrix(newvs)
	
	i = 1
	while i < msize:
		pivot = i
		j = i
		while j < msize:
			if abs(mat[(j,i)]) > abs(mat[(pivot,i)]):
				pivot = j
			j += 1
	
		if abs(mat[(i,i)]) < eps:
			return Vector([]) #singular

		mat.swap(i, pivot)
	
		k = i+1
		while k < msize:
			mat[(i,k)] /= mat[(i,i)] #TODO:ERROR
			k += 1
			
		for idx in range(msize):
			if i != idx:
				l = i+1
				while l <= msize:
					mat[(idx,l)] -= 	mat[(idx,i)]*mat[(i,l)]
					
		i += 1
		
	return mat[msize-1]
	
#Entry point
if __name__ == '__main__':
	print("linear module load")
		
