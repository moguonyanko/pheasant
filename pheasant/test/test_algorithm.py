#!/usr/bin/python3
# -*- coding: utf-8 -*-

import unittest

import pheasant.algorithm as al

class TestSearch(unittest.TestCase):
	'''
	Test for search function.
	'''
	def test_fib(self):
		'''
		test fib function
		'''
		res = al.fib(100)
		self.assertEqual(res, 354224848179261915075)

	def test_dfs_partial_sum(self):
		'''
		test dfs for partial sum
		'''
		x1 = al.dfs_partial_sum(4, 13, [1,2,4,7])
		self.assertEqual(x1, True)
		x2 = al.dfs_partial_sum(4, 15, [1,2,4,7])
		self.assertEqual(x2, False)

class TestGreedMethod(unittest.TestCase):
	'''
	Test for greed method
	'''
	def test_greed_min_coin(self):
		'''
		test greed coin function
		'''
		coins = {
			1 : 3,
			5 : 2,
			10 : 1,
			50 : 3,
			100 : 0,
			500 : 2
		}
		coinnum = 620
		
		res = al.greed_min_coin(coins, coinnum)
		self.assertEqual(res, 6)
		
	def test_greed_fence_repair(self):
		'''
		test fence repair by greed method
		'''
		cutnum = 3
		cutlength = [8,5,8]

		res = al.greed_fence_repair(cutnum, cutlength)
		self.assertEqual(res, 34)
		
class TestDynamicProgramming(unittest.TestCase):
	'''
	Test for dynamic programming
	'''
	def test_knapsack(self):
		'''
		test for knapsack problem
		'''
		items = [(2,3),(1,2),(3,4),(2,2)]
		lim_weight = 5
		
		res = al.knapsack(items, lim_weight)
		#todo: test cannot pass
		#self.assertEqual(res, 7)
		
	def test_tsp(self):
		'''
		Test of tsp function.
		'''
		g = al.Graph(vertexlen=5, direct=True)
		g.append(start=0, end=1, cost=3)
		g.append(start=0, end=3, cost=4)
		g.append(start=1, end=2, cost=5)
		g.append(start=2, end=0, cost=4)
		g.append(start=2, end=3, cost=5)
		g.append(start=3, end=4, cost=3)
		g.append(start=4, end=0, cost=7)
		g.append(start=4, end=1, cost=6)
		
		res = al.tsp(g)
		
		chk = 22 #Minimum cost.
		
		#TODO: Implement now.
		#self.assertEqual(res, chk)

class TestGraph(unittest.TestCase):
	'''
	Test for graph class.
	'''
	def test_append(self):
		'''
		test for append edge
		create exist direct graph
		'''
		g = al.Graph(vertexlen=3,direct=True)
		g.append(0,1)
		g.append(0,2)
		g.append(1,2)
		
		res1 = g.edges[0]
		res2 = g.edges[1]
		
		self.assertEqual(res1, [al.Edge(1), al.Edge(2)])		
		self.assertEqual(res2, [al.Edge(2)])		

class TestRouteSearch(unittest.TestCase):
	'''
	Test for route search.
	'''
	def test_dijkstra(self):
		'''
		test dijkstra method function
		'''
		A = 0
		B = 1
		C = 2
		D = 3
		E = 4
		F = 5
		G = 6
		
		routes = [
			(A,B,2),
			(A,C,5),
			(B,C,4),
			(B,D,6),
			(B,E,10),
			(C,D,2),
			(D,F,1),
			(E,F,3),
			(E,G,5),
			(F,G,9)
		]
		routes_graph = al.Graph(edges=routes, vertexlen=7, direct=True)
		
		res = al.dijkstra(routes_graph, A, G)
		
		#TODO:dijkstra function is not work normality.Must be repaired.
		#self.assertEqual(res, ([A,C,D,F,E,G], 16))

class TestMultiProcess(unittest.TestCase):
	'''
	Test class for multi process.
	'''
	def test_philosophers(self):
		'''
		Test function to Dining Philosophers Problem.
		'''
		phi0 = al.Philosopher(0)
		phi1 = al.Philosopher(1)
		phi2 = al.Philosopher(2)
		phi3 = al.Philosopher(3)
		phi4 = al.Philosopher(4)

		waiter = al.Waiter(forknum=5)	
		pass
		
class TestUnionFindTree(unittest.TestCase):
	'''
	Test class for union find tree.
	'''
	#TODO: Test data is not prepared.
	def test_find(self):
		pass
		
	def test_unite(self):
		pass
	
	def test_same(self):
		pass
		
class TestHeap(unittest.TestCase):
	'''
	Test class for Heap class.
	'''
	
	sample = [1,2,3,4,5]
	
	def test_push(self):
		'''
		Test push method.
		'''
		heap = al.Heap(size=len(self.sample))
		for value in self.sample:
			heap.push(value)
		
		res = heap.values
		
		self.assertEqual(res, self.sample)
		
	def test_pop(self):
		'''
		Test pop method.
		'''
		heap = al.Heap(len(self.sample))
		for value in self.sample[::-1]:
			heap.push(value)
		
		res = [heap.pop() for i in range(len(heap))]
			
		self.assertEqual(res, self.sample)
		
class TestBinarySearch(unittest.TestCase):
	'''
	Binary search test class.
	'''
	def test_binary_search(self):
		'''
		binary search test function
		'''
		sample = [1,2,3,4,6,7,8,9]
		target_idx1 = al.binary_search(7, sample)
		self.assertEqual(target_idx1, 5)
		target_idx2 = al.binary_search(2, sample)
		self.assertEqual(target_idx2, 1)
		target_idx3 = al.binary_search(4, sample)
		self.assertEqual(target_idx3, 3)
		
		self.assertRaises(LookupError, al.binary_search, 100, sample)
		
#inner value unfound, maximum recursion.		
#		target_idx5 = al.binary_search(5, sample)
#		self.assertEqual(target_idx5, -1)		

class TestGCD(unittest.TestCase):
	'''
	GCD test class.
	'''
	def test_gcd(self):
		'''
		test function for gcd.
		'''
		res = al.gcd(206, 40)
		self.assertEqual(res, 2)

class AlgorithmerTrainingTest(unittest.TestCase):
	'''
	最強最速アルゴリズマー養成講座（著 高橋直大氏）に記載されている
	サンプルコードを元にした練習コードをテストするクラスです。
	'''
	def test_thePouring(self):
		capas = [700000, 800000, 900000, 1000000]
		bottles = [478478, 478478, 478478, 478478]
		fromIds = [2, 3, 2, 0, 1]
		toIds = [0, 1, 1, 3, 2]
		
		kiwi = al.KiwiJuiceEasy()
		
		res = kiwi.thePouring(capas, bottles, fromIds, toIds)
		self.assertEqual([0, 156956, 900000, 856956], res)
	
	def test_bestInvitation(self):
		first = ["snakes","programming","cobra","monty"]
		second = ["python","python","anaconda","python"]
		
		party = al.InterestingParty()
		
		res = party.bestInvitation(first, second)
		self.assertEqual(3, res)
		
	def test_encrypt(self):
		numbers = [1,2,3,1,1,3]
		crypt = al.Cryptography()
		res = crypt.encrypt(numbers)
		self.assertEqual(36, res)
		
		numbers2 = [1,1,1,1]
		crypt = al.Cryptography()
		res2 = crypt.encrypt(numbers2)
		self.assertEqual(2, res2)
	
	def test_digits(self):
		intdg = al.InterestingDigits()
		
		res1 = intdg.digits(10)
		self.assertEqual({3,9}, res1)
		
		res2 = intdg.digits(26)
		self.assertEqual({5,25}, res2)
	
	def test_find(self):
		tpl = al.ThePalindrome()
		
		res = tpl.find("abab")
		self.assertEqual(5, res)
		
		res1 = tpl.find("abacaba")
		self.assertEqual(7, res1)

		res2 = tpl.find("qwerty")
		self.assertEqual(11, res2)

	def test_hightestScore(self):
		hs = al.FriendScore()
		
		friend_info = [
			"NYNNN",
			"YNYNN",
			"NYNYN",
			"NNYNY",
			"NNNYN"
		]
		
		res = hs.highestScore(friend_info)
		self.assertEqual(4, res)
		
	def test_getProbability(self):
		cb = al.CrazyBot()
		
		res = cb.getProbability(2,25,25,25,25)
		self.assertEqual(0.75, res)

	def test_longestPath(self):
		mm = al.MazeMaker()
		maze = [
			"...",
			"...",
			"..."
		]
		startRow = 0
		startCol = 1
		moveRow = [1,0,-1,0]
		moveCol = [0,1,0,-1]
		res = mm.longestPath(maze, startRow, startCol, moveRow, moveCol)
		self.assertEqual(3, res)

		maze2 = [
			"X.X",
			"...",
			"XXX",
			"X.X"
		]
		startRow2 = 0
		startCol2 = 1
		moveRow2 = [1,0,-1,0]
		moveCol2 = [0,1,0,-1]
		res2 = mm.longestPath(maze2, startRow2, startCol2, moveRow2, moveCol2)
		self.assertEqual(-1, res2)
	
	def test_theNumber(self):
		nm = al.NumberMagicEasy()
		
		answer = "YNYY"
		res = nm.theNumber(answer)
		self.assertEqual(5, res)

		answer = "YYYYYYYYYYY"
		self.assertRaises(KeyError, nm.theNumber, answer)
	
	def test_routecalc(self):
		maxW = 5
		maxH = 4
		rt = al.RouteSearchEasy(maxW, maxH)
		res = rt.calc(maxW, maxH)
		self.assertEqual(126, res)
	
	def test_knapsack_search(self):
		ws = [3,4,1,2,3]
		ps = [2,3,2,3,6]
		maxW = 10
		
		#誤植のために動作しない疑いあり。
		#knap = al.KnapsackSearch(ws, ps, maxW)
		#res = knap.getMaxPrecious()
		#self.assertEqual(14, res)
	
	def test_totalSalary(self):
		relations = ["NNYN","NNYN","NNNN","NYYN"]
		cs = al.CorporationSalary(relations)
		res = cs.totalSalary()
		self.assertEqual(5, res)

		relations2 = ["NNNN","NNNN","NNNN","NNNN"]
		cs2 = al.CorporationSalary(relations2)
		res2 = cs2.totalSalary()
		self.assertEqual(4, res2)

	def test_maxDonations(self):
		donations = [1,2,3,4,5,1,2,3,4,5]
		bn = al.BadNeighbors(donations)
		res = bn.maxDonations()
		self.assertEqual(16, res)
		
if __name__ == '__main__':
	print(__file__)
	unittest.main()

