#!/usr/bin/python3
# -*- coding: utf-8 -*-

import random
import math
import queue as qu

import pheasant.util as ut

def fib(number):
	'''
	fibonach function
	'''
	memo = ut.makelist(number+1,0)
	
	def rec_fib(n):
		if n <= 1:
			return n
		if memo[n] != 0 :
			return memo[n]
		memo[n] = rec_fib(n-1)+rec_fib(n-2)
		return memo[n]
		
	return rec_fib(number)

def dfs_partial_sum(maxnum, result, nums):
	'''
	Depth First Search for patial sum
	'''
	def rec_dfs_partial_sum(i, sumnum):
		if i == maxnum:
			return sumnum == result
		if rec_dfs_partial_sum(i+1, sumnum):
			return True
		if rec_dfs_partial_sum(i+1, sumnum+nums[i]):
			return True
		return False
	
	return rec_dfs_partial_sum(0, 0)

def greed_min_coin(coins, target):
	'''
	minimum coin use
	'''
	total_coin_sheets = 0
	rev_coin_keys = sorted(coins.keys(), reverse=True)

	for coin in rev_coin_keys:
		coin_sheets = min(math.floor(target/coin), coins[coin])
		target -= coin_sheets*coin
		total_coin_sheets += coin_sheets
	
	return total_coin_sheets

def greed_fence_repair(cutnum, cut_length_values):
	'''
	Fence repair by greed method
	'''
	cost = 0
	
	while cutnum > 1:
		first_cut_idx = 0
		second_cut_idx = 1
		
		if cut_length_values[first_cut_idx] > cut_length_values[second_cut_idx]:
			first_cut_idx, second_cut_idx = ut.swap(first_cut_idx, second_cut_idx)
		
		i = 2
		while True:
			if i < cutnum:
				if cut_length_values[i] < cut_length_values[first_cut_idx]:
					second_cut_idx = first_cut_idx
					first_cut_idx = i
				elif cut_length_values[i] < cut_length_values[second_cut_idx]:
					second_cut_idx = i
				i += 1
			else:
				break
			
		now_cost = cut_length_values[first_cut_idx]+cut_length_values[second_cut_idx]
		cost += now_cost

		if first_cut_idx == cutnum-1:
			first_cut_idx, second_cut_idx = ut.swap(first_cut_idx, second_cut_idx)
			
		cut_length_values[first_cut_idx] = now_cost
		cut_length_values[second_cut_idx] = cut_length_values[cutnum-1]
		cutnum -= 1
	
	return cost

def knapsack(items, lim_weight):
	'''
	knapsack problem by dynamic programming
	'''
	#TODO:Uncorrect value returned.
	item_len = len(items)
	weights = []
	values = []
	for w, v in items:
		weights.append(w)
		values.append(v)
	dp = [[0]*(lim_weight+1)]*(item_len+1)
	
	i = item_len-1
	while i >= 0:
		j = 0
		while j <= lim_weight:
			if j < weights[i]:
				dp[i][j] = dp[i+1][j]
			else:
				dp[i][j] = max(dp[i+1][j], dp[i+1][j-weights[i]]+values[i])
			j += 1
		i -= 1
	
	return dp[0][lim_weight]

class Edge():
	'''
	Edge is parts of graph algorithm.
	'''
	def __init__(self, to, cost=float("inf")):
		'''
		United vertex number and cost initialize.
		to : append united vertex number.
		cost : route length, money...
		'''
		self.to = to
		self.cost = cost

	def __str__(self):
		'''
		All united vertex return.
		'''
		return str(self.to)
		
	def __eq__(self, target):
		'''
		If to and cot equal, return True.
		'''
		return self.to == target.to and self.cost == target.cost
		
class Vertex():
	'''
	Vertex of graph. 
	'''
	def __init__(self, cost=0):
		'''
		Required cost.
		'''
		self.cost = cost
		self.vertexes = []
		
	def __str__(self):
		'''
		All united vertex return.
		'''
		return str(self.vertexes)
	
	def append(self, vtx):
		'''
		Append united vertex.
		'''
		self.vertexes.append(vtx)

class Graph():
	'''
	Graph class
	It is used by route search, geometry etc.
	'''	
	def __init__(self, edges=[], vertexlen=0, direct=True):
		'''
		Vertex length and edge length require.
		Direct false, none direct graph express.
		Default direct is True, assume DAG.
		Edges is tuples list.
		'''
		self.vertexlen = vertexlen
		self.edgelen = len(edges)
		self.direct = direct
		self.edges = {}
		for start, end, cost in edges:
			self.append(start, end, cost)

	def __str__(self):
		'''
		All included vertexes return.
		'''
		return str(self.edges)
		
	def edgelen():
		'''
		Edge length utillity.
		'''		
		return len(self.edges)

	def append(self, start, end, cost=float("inf")):
		'''
		Append vertex.
		'''
		if not start in self.edges:
			self.edges[start] = []
		self.edges[start].append(Edge(end, cost))
		if self.direct == False:
			if not end in self.edges:
				self.edges[end] = []
			self.edges[end].append(self.edges[start]) #If cost not equal, Edge must be initialized.

def dijkstra(route, start, end):
	'''
	Dijkstra route search function with priority queue.
	Return tuple of most smallest route list and cost.
	Argument start is start vertex index.
	Argument end is end vertex index.
	'''
	#TODO:This is not work normality.Must be repaired.
	que = qu.PriorityQueue()
	vtxlen = route.vertexlen
	distance = ut.makelist(length=vtxlen, initvalue=float("inf"))
	distance[start] = 0
	que.put((0, start)) #shorest distance, vertex number
	
	smallroute = [start]
	totalcost = 0
	while que.empty() == False:
		smldist = que.get()
		vertexid = smldist[1]
		if distance[vertexid] < smldist[0]:
			continue
		
		for vtxidx in route.edges:
			edges = route.edges[vtxidx]
			for edgidx in range(len(edges)):
				edge = edges[edgidx]
				cost = distance[vertexid]+edge.cost
				if distance[edge.to] > cost:
					distance[edge.to] = cost
					que.put((distance[edge.to], edge.to))
					smallroute.append(edge.to)
					totalcost += edge.cost
						
	return (smallroute, totalcost)
	
def without_priority_queue_dijkstra(route, start, end):
	'''
	Dijkstra route search function 
	without priority queue.
	'''	
	#TODO:This is not work normality.Must be repaired.
	vtxlen = route.vertexlen
	d = ut.makelist(length=vtxlen, initvalue=float("inf"))
	used = ut.makelist(length=vtxlen, initvalue=False)
	d[start] = 0
	prev = ut.makelist(length=vtxlen, initvalue=-1)

	while True:
		v = -1
		for u in range(vtxlen):
			if (used[u] == False) and (v == -1 or d[u] < d[v]):
				v = u
				
		if v == -1:
			break
		
		used[v] = True
		
		for u in range(vtxlen):
			cost = d[v]+route.edges[v][u].cost
			if d[u] > cost:
				d[u] = cost
				prev[u] = v
				
	path = []
	t = end
	while t != -1:
		t = prev[t]
		path.append(t)
	
	return (path[::-1], d[end])

#Dining Philosophers Problem	
class Waiter():
	'''
	Dining Philosophers Problem.
	This class express waiter.
	Waiter deal forks	on the table.
	'''
	def __init__(self, forknum=5):	
		self.forks = ut.makelist(length=forknum, initvalue=False)
		
	def __getsuffix(phinum, side):
		suffix = None
		if side == "right":
			suffix = phinum
		else:
			suffix = (phinum+1)%forknum
			
		return suffix
		
	def existfork(self, phinum, side):
		'''
		phinum: Philosopher number.
		side: Fork side.
		'''
		suffix = __getsuffix(phinum, side)
		return self.forks[suffix]
	
	def forkset(self, phinum, side, exist):
		'''
		Fork exist update.
		phinum: Philosopher number.
		side: Fork side.
		exist: Bool of fork exist or not exist.
		'''
		suffix = __getsuffix(phinum, side)
		self.forks[suffix] = exist
		
	def getfork(self):
		'''
		Get fork on the table.
		'''
		#TODO: implement
		pass

	def putfork(self):
		'''
		Put fork on the table.
		'''
		#TODO: implement
		pass

class Philosopher():
	'''
	Dining Philosophers Problem.
	This class express philosopher.
	'''
	def __init__(self, number):
		'''
		Philosopher initialize.
		number: Number of philosopher.
		'''
		self.number = number
		
	def eat(self, table):
		'''
		Philosopher's action of eating.
		'''
		eatlim = 2
		eatnum = 0
		while eatnum <= eatlim:
			if eatnum == eatlim:
				print(self.number + " is sleeping...")
			else:
				print(self.number + "is thinking...")
				table.getfork(number, "right")
				table.getfork(number, "left")
				print(self.number + "is eating...")
				yield number
				table.putfork(number, "right")
				table.putfork(number, "left")
				eatnum += 1

class UnionFindTree():
	'''
	Union find tree.
	'''
	def __init__(self, elenum):
		'''
		Initialize by element number.
		'''
		for i in range(elenum):
			self.par[i] = i
			self.rank[i] = 0
	
	def find(self, target):
		'''
		Find target root.
		'''
		if self.par[target] == target:
			return target
		else:
			self.par[target] = self.find(par[target])
			return self.par[target]
			
	def unite(self, a, b):
		'''
		Unite a and b set.
		'''
		a = self.find(a)
		b = self.find(b)
		if a == b:
			return
		
		if self.rank[a] < self.rank[b]:
			self.par[a] = b
		else:
			self.par[b] = a
			if self.rank[a] == self.rank[b]:
				self.rank[a] += 1
		
	def same(self, a, b):
		'''
		Is a and b same set?
		'''
		return self.find(a) == self.find(b)

class Heap():
	'''
	Heap data structure class.
	'''
	error_message = "Requested value is out of bounds."
	
	def __init__(self, size=0, values=[]):
		'''
		Init heap with heap size and values.
		'''
		if len(values) > 0:
			self.values = values
		else:
			self.values = ut.makelist(size)
		
		self.pos = 0
		
	def __len__(self):
		'''
		Heap length return.
		'''
		return len(self.values)
	
	def push(self, value):
		'''
		Push to heap.
		'''
		if len(self.values) <= self.pos: 
			raise ValueError(self.error_message)
		
		i = self.pos
		self.pos += 1
		
		while i > 0:
			idx = int((i-1)/2)
			if self.values[idx] <= value:	break
			
			self.values[i] = self.values[idx]
			i = idx
		
		self.values[i] = value
		
	def pop(self):
		'''
		Pop from heap.
		'''
		if self.pos <= 0:	
			raise ValueError(self.error_message)
		
		ret = self.values[0]
		
		self.pos -= 1
		x = self.values[self.pos]
		
		i = 0
		while i*2+1 < self.pos:
			l = i*2+1;r = i*2+2
			
			if r < self.pos and self.values[r] < self.values[l]:
				l = r
				
			if self.values[l] >= x:	break 
				
			self.values[i] = self.values[l]
			i = l
		
		self.values[i] = x
		
		return ret
		
def binary_search(target, samples):
	'''
	binary search by recursive
	'''
	#TODO:Inner value unfound, maximum recursion.
	if target > max(samples) or target < min(samples):
		raise LookupError("Not found target value.")
		
	def rec_search(left, right):
		if left >= right:
			raise LookupError("Not found target value.")
		pivot = math.floor((left+right)/2)
		if target == samples[pivot]:
			return pivot
		elif target > samples[pivot]:
			return rec_search(pivot, right)
		elif target < samples[pivot]:
			return rec_search(left, pivot)
	
	return rec_search(0, len(samples)-1)
	
def normal_binary_search(target, samples):
	'''
	normal binary search
	'''
	if target > max(samples) or target < min(samples):
		raise LookupError("Requested value is out of range.")
		
	left = 0
	right = len(samples)-1
	while left <= right:
		middle = math.floor((left+right)/2)
		if target == samples[middle]:
			return middle
		elif target < samples[middle]: 
			right = middle
		elif samples[middle] < target:
			left = middle
	
	raise LookupError("Not found target value.")		
	
def gcd(a, b):
	'''
	Calculate gcd.
	'''
	#TODO:If a or b is negative, value is undefined.
	if b == 0:
		return a
	else:
		x = abs(a)%abs(b)
		if (not (a<0 and b<0)) and (a<0 or b<0):
			x *= -1
		return gcd(b, x)	
	
def tsp(rts):
	'''
	Traveling Salesman Probrem.
	rts: Routes by graph expression.
	'''
	#TODO: Implement now.
	pass

##
# 最強最速アルゴリズマー養成講座（著 高橋直大氏）を参考にした，
# Pythonによる練習コード
##

class KiwiJuiceEasy():
	'''
	P.53 シミュレーション
	'''
	def thePouring(self, capas, bottles, fromIds, toIds):
		size = len(fromIds)
		for i in range(size):
			fromId = fromIds[i]
			toId = toIds[i]
		
			move_vol = min(bottles[fromId], capas[toId]-bottles[toId])

			bottles[fromId] -= move_vol
			bottles[toId] += move_vol
	
		return bottles

class InterestingParty():
	'''
	P.71 全探索
	'''
	def bestInvitation(self, first, second):
		dic = {}
		rng = range(len(first))
		
		for i in rng:
			dic[first[i]] = 0	
			dic[second[i]] = 0	
		
		for i in rng:
			dic[first[i]] = dic[first[i]]+1	
			dic[second[i]] = dic[second[i]]+1
		
		ans = 0
		for key in dic:
			if ans < dic[key]: ans = dic[key]
	
		return ans

class Cryptography():
	'''
	P.77 全探索
	'''
	def encrypt_V1(self, numbers):
		ans = 0
		rng = range(len(numbers))
		for i in rng:
			seki = 1
			for j in rng:
				if i == j: #自分のターンなら自分に1足して掛ける。つまりここは1足す項を探すためのループ。
					seki *= (numbers[j]+1)
				else:
					seki *= numbers[j]
			
			ans = max(ans, seki)
		
		return ans

	def encrypt(self, numbers):
		rng = range(len(numbers))
		nums = sorted(numbers)
		seki = 1
		nums[0] = nums[0]+1 #最小値+1が最大の増加率を導く。
		for i in rng: seki *= nums[i]
		
		return seki

class InterestingDigits():
	'''
	P.88 全探索
	'''
	def digits(self, base):
		res = set()
		rng = range(base)
		
		n = 2
		while n < base:
			ok = True
			for k1 in rng:
				for k2 in rng:
					for k3 in rng:
						#10進数表記に変換して，nの倍数かどうかと各桁の和がnの倍数でないかどうかを確認する。
						if (k1 + k2*base + k3*base*base) % n == 0 and (k1 + k2 + k3) % n != 0:
							ok = False
							break
					if ok == False: break
				if ok == False: break
			if ok == True: res.add(n)
			n += 1
			
		return res

	def digits_V2(self, base):
		'''
		求めるnはbaseを法として1と合同な数字である。
		従って，
			1≡n(mod base)
		ここで左辺を右辺に移行して
			0≡n-1(mod base)
		となる。n-1はbaseを法として0と合同なので
		n-1はbaseの倍数だと分かる。
		'''
		res = set()
		n = 2
		while n < base:
			if (base-1) % n == 0: res.add(n) #TODO: 何故nで割るのか？
			n += 1
		
		return res

class ThePalindrome():
	'''
	P.95 全探索
	回文になる場合の，最小の文字数を返します。
	'''
	def find(self, s):
		size = len(s)
		rng = range(size)
		
		for i in rng:
			matchflag = True
			for j in rng:
				opposite_index = i - j - 1
				#比較する文字を1つずつずらすことで「1文字追加される」という処理を表現している。
				#「1文字追加される」ことで反対側の文字が存在するようになる。
				if opposite_index < size and s[j] != s[opposite_index]:
					matchflag = False
					break
			
			if matchflag == True: return i+size
		
		return size*2-1 #両端から突き合わせたが全く一致しなかった。最長の回文の文字数を返す。

class FriendScore():
	'''
	P.104
	全探索
	'''
	def highestScore_V1(self, friends):
		rng = range(len(friends[0]))
		friend_key = "Y"
		
		max_friend_count = 0
		for i in rng:
			cnt = 0
			
			for j in rng:
				if i == j: continue #自分自身を友人として数えない。
				
				if friends[i][j] == friend_key:
					cnt += 1
				else:
					for k in rng:
						if friends[j][k] == friend_key and friends[k][i] == friend_key:
							cnt += 1
							break
							
			max_friend_count = max(max_friend_count, cnt)
		
		return max_friend_count

	def highestScore(self, friends):
		is_friends = "Y"
		
		max_friend_count = 0
		for i, user_i in enumerate(friends):
			cnt = 0
			
			for j, user_j in enumerate(friends):
				if user_i == user_j: continue #自分自身を友人として数えない。
				
				if user_i[j] == is_friends:
					cnt += 1
				else:
					for k, user_k in enumerate(friends):
						if user_i[k] == user_k[i] == is_friends:
							cnt += 1
							break
							
			max_friend_count = max(max_friend_count, cnt)
		
		return max_friend_count

class CrazyBot():
	'''
	P.129 全探索
	'''
	def __init__(self):
		self.direct = 4
		capacity = 100
	
		self.grid = {}
		for column_num in range(capacity): #正方形という設定
			column = [False]*capacity
			self.grid[column_num] = column
		
		self.vx = [1, -1, 0, 0]
		self.vy = [0, 0, 1, -1]
		self.prob = [0.0]*self.direct
		
	def getProbability(self, n, east, west, south, north):
		directs = [east, west, south, north]

		for i, d in enumerate(directs):
			self.prob[i] = d/100.0
		
		firstx = firsty = 50
		
		return self.__dfs(firstx, firsty, n)
	
	def __dfs(self, x, y, n):
		if self.grid[x][y] == True: return 0 #1度通った場所なので確率0を返す。
		if n == 0: return 1 #もう歩けない。TODO: ここで1を返して良い理屈は？
		
		self.grid[x][y] = True
		ret = 0
		
		for i in range(self.direct):#4方向にロボットを動かしてそれぞれの確率を足し込んでいく。
			ret += self.__dfs(x+self.vx[i], y+self.vy[i], n-1) * self.prob[i]
		
		self.grid[x][y] = False #別のルート探索で通れるよう，より浅いところに上がってくるときにフラグを戻しておく。
		
		return ret
		
class MazeMaker():
	'''
	P.139 全探索
	'''
	def isStep(self, nextX, nextY, width, height, board, maze):
		'''
		Check valid step.
		'''
		if 0 <= nextX and nextX < width and 0 <= nextY and nextY < height and board[nextY][nextX] == -1 and maze[nextY][nextX] == ".": 	
			return False
		
		return True

	def longestPath(self, maze, startRow, startCol, moveRow, moveCol):
		width = len(maze[0])
		height = len(maze)
		board = ut.makeArray(height, width, initValue=-1)
		
		board[startRow][startCol] = 0
		
		queueX = qu.Queue()
		queueY = qu.Queue()
		queueX.put(startCol)
		queueY.put(startRow)
		
		rowrng = range(len(moveRow))
		while queueX.empty() == False:
			x = queueX.get()
			y = queueY.get()
		
			for i in rowrng:
				nextX = x + moveCol[i]
				nextY = y + moveRow[i]
				
				if 0 <= nextX and nextX < width and 0 <= nextY and \
				nextY < height and board[nextY][nextX] == -1 and maze[nextY][nextX] == ".": 	
					board[nextY][nextX] = board[y][x] + 1 #有効なステップならば歩数として数えてboradに追加する。
					queueX.put(nextX)
					queueY.put(nextY)
			
		max_step = 0

		for i in range(height):
			for j in range(width):
				if maze[i][j] == "." and board[i][j] == -1: #通れるが−1，つまり到達できないマスがあった。
					return -1
				max_step = max(max_step, board[i][j])			
		
		return max_step

class NumberMagicEasy():
	'''
	P.151 全探索
	'''
	def __init__(self):
		limit = 16
		soldict = {}
		solvalues = [
			"YYYY",
			"YYYN",
			"YYNY",
			"YYNN",
			"YNYY",
			"YNYN",
			"YNNY",
			"YNNN",
			"NYYY",
			"NYYN",
			"NYNY",
			"NYNN",
			"NNYY",
			"NNYN",
			"NNNY",
			"NNNN"
		]
		
		for i in range(limit):
			soldict[solvalues[i]] = i+1
		
		self.solution = soldict
	
	def theNumber(self, answer):
		try:
			return self.solution[answer]
		except KeyError as ex:
			raise KeyError("Cannot find your number.")
			
class RouteSearchEasy():
	'''
	P.177 動的計画法・メモ化
	'''
	def __init__(self, h, w):
		sizeW = w+1
		sizeH = h+1
		dp = ut.makeArray(sizeH, sizeW, 0)
		dp[0][0] = 1
		for i in range(sizeH):
			for j in range(sizeW): 
				if i != 0: #上に進む
					dp[i][j] += dp[i-1][j]
				if j != 0: #右に進む
					dp[i][j] += dp[i][j-1]
		
		self.dp = dp
	
	def calc(self, h, w):
		return self.dp[h][w]

class KnapsackSearch():
	'''
	P.184 動的計画法・メモ化
	'''
	def __init__(self, ws, ps, maxW):
		weightSize = len(ws)+1
		maxSize = maxW+1
		dp = ut.makeArray(weightSize, maxSize, 0)
		
		ret = 0
		for i in range(weightSize):
			for j in range(maxSize):
				if j+ws[i] <= maxW:
					dp[i+1][j+ws[i]] = max(dp[i+1][j+ws[i]], dp[i][j] + ps[j]) #ps[j]は確実にエラー。誤植？
					ret = max(dp[i+1][j+ws[i]], ret)
		
		self.maxPrecious = ret
		self.dp = dp
	
	def getMaxPrecious(self):
		return self.maxPrecious

class CorporationSalary():
	'''
	P.193 動的計画法・メモ化
	'''
	def __init__(self, relations):
		self.relations = relations
		self.salaries = [0]*len(self.relations)

	def totalSalary(self):
		total = 0
		for i, emp in enumerate(self.relations):
			total += self.__getSalary(i)

		return total

	def __getSalary(self, i):
		if self.salaries[i] == 0:
			salary = 0
			relation = self.relations[i]
			
			for j, char in enumerate(relation):
				if relation[j] == "Y":
					salary += self.__getSalary(j)
			
			if salary == 0: salary = 1
			
			self.salaries[i] = salary
		
		return self.salaries[i]
	
class BadNeighbors():
	'''
	P.204 動的計画法・メモ化
	'''
	def __init__(self, donations):
		self.donations = donations
		self.dp = [0]*len(donations)

	def __calcDonation(self, donation, i, ans):
		self.dp[i] = donation
		if i > 0:
			self.dp[i] = max(self.dp[i], self.dp[i-1])
		if i > 1:
			self.dp[i] = max(self.dp[i], self.dp[i-2]+donation)
		
		return max(ans, self.dp[i])
	
	def maxDonations(self):
		ans0 = 0
		ans1 = 0
	
		for i in range(len(self.donations)-1):
			ans0 = self.__calcDonation(self.donations[i], i, ans0)
			ans1 = self.__calcDonation(self.donations[i+1], i, ans1)
				
		return max(ans0, ans1)

class ChessMetric():
	'''
	P.213 動的計画法・メモ化
	'''
	def __init__(self, boardsize, x_range, y_range, max_movenum=55):
		self.size = boardsize
		ways = ut.makeArray(boardsize, boardsize, 0)	
		for i in range(len(ways)):
			for j in range(len(ways[i])):
				ways[i][j] = [0]*max_movenum
		self.ways = ways
		self.dx = x_range
		self.dy = y_range

	def howMany(self, start, end, numMoves):
		if numMoves > len(self.ways[0][0]):
			raise ValueError("Exceed max move limit!")
	
		sx = start[0]
		sy = start[1]
		ex = end[0]
		ey = end[1]
		
		self.ways[sy][sx][0] = 1 #スタート地点の到達パターン数は1通り
		boardrange = range(self.size) 
		wayrange = range(len(self.dx))

		for i in range(numMoves+1):
			i += 1
			for x in boardrange:
				for y in boardrange:
					for j in wayrange:
						nx = x + self.dx[j]
						ny = y + self.dy[j]
						
						if nx < 0 or ny < 0 or nx >= self.size or ny >= self.size: 
							continue
						else: #1つ前のマスまでの到達パターン数を足し込む。
							self.ways[ny][nx][i] += self.ways[y][x][i-1]
		
		return self.ways[ex][ey][numMoves]
	
#Entry point
if __name__ == '__main__':
	print("algorithm module load")

