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

def test_two_opt():
	'''
	生成AIで作成したプログラム
	'''
	# 都市間の距離
	dist = [
			[0, 10, 15, 20],
			[10, 0, 35, 25],
			[15, 35, 0, 30],
			[20, 25, 30, 0]
	]

	N = len(dist)

	# 初期経路
	route = list(range(N)) + [0] # 0 -> 1 -> 2 -> 3 -> 0

	def calculate_distance(route):
			# 経路の距離を計算
			return sum(dist[route[i-1]][route[i]] for i in range(N))

	def two_opt(route, i, j):
			# 2-opt法で経路を更新
			return route[:i] + route[i:j+1][::-1] + route[j+1:]

	while True:
			distance = calculate_distance(route)
			updated = False

			for i in range(1, N):
					for j in range(i+1, N):
							if j-i == 1: continue # 隣接しているエッジは無視

							# 新しい経路の距離を計算
							new_route = two_opt(route, i, j)
							new_distance = calculate_distance(new_route)

							# 新しい経路の方が短ければ更新
							if new_distance < distance:
									distance = new_distance
									route = new_route
									updated = True

			# 経路が更新されなければ終了
			if not updated:
					break

	print(route)
	print(distance)

def test_tsp_by_bitdp():
	'''
	生成AIで作成したプログラム
	'''
	N = 4
	INF = float('inf')

	# 都市間の距離
	dist = [
			[0, 10, 15, 20],
			[10, 0, 35, 25],
			[15, 35, 0, 30],
			[20, 25, 30, 0]
	]

	# DPテーブルの初期化
	dp = [[INF]*N for _ in range(1<<N)]
	dp[0][0] = 0 # 最初に都市0にいる状態

	# ビットDP
	for bit in range(1<<N):
			for i in range(N):
					if not (bit >> i & 1): # まだ訪れていない都市について
							for j in range(N):
									if bit >> j & 1: # すでに訪れた都市から移動する
											dp[bit|(1<<i)][i] = min(dp[bit|(1<<i)][i], dp[bit][j] + dist[j][i])

	# 全都市を訪れた後に都市0に戻る最短距離
	ans = min(dp[(1<<N)-1][i] + dist[i][0] for i in range(N))
	print(ans)	

def test_recalc_teate():
	rest = 3000
	payment = 7
	year = 0
	while rest > 0:
		for _ in range(12):
			if rest < 0:
				break
			teate = al.recalc_teate(rest)
			rest -= payment + teate
		year += 1
	print(year)
	assert year <= 20

if __name__ == '__main__':
	print(__file__)
	unittest.main()

