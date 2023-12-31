# -- coding: utf-8 --**
# @author zjp
# @time 22-05-23_19.42.53
## 获取某一节点到其他节点的最短依存路径
import numpy as np 
class get_adj_sdp:
	'''为使用邻接矩阵表示表示的图实现 Dijkstra 的单源最短路径算法'''   
	def __init__(self,graph):
		self.graph = np.array(graph)
	
	def minDistance(self,dist,queue):
		'''从仍在队列中queue的顶点集中找到具有最小距离值的顶点的函数'''
		minimum = 1e+19		# 将minimum 和 min_index 初始化
		min_index = -1		
		for i in range(len(dist)):		# 从 dist 数组中，选择一个具有最小值并且在队列中的节点
			if dist[i] < minimum and i in queue:
				minimum = dist[i]
				min_index = i
		return min_index
	
	def printPath(self, parent, j):	
		'''使用父数组打印从src到 j 的最短路径'''		
		if parent[j] == -1 :
			print(j,end=" ")
			return
		self.printPath(parent , parent[j])
		print (j,end=" ")

	def get_sdp(self, sdp, parent, j):			
		'''返回最短依存路径'''
		if parent[j] == -1 :
			sdp.append(j)
			return sdp
		self.get_sdp(sdp, parent , parent[j])
		sdp.append(j)
		return sdp 

	def printSolution(self, src, dist, parent):
		'''打印最短依存路径'''
		print("Vertex \t\tDistance from Source\tPath")
		for i in range(0, len(dist)):
			print("\n%d --> %d \t\t%d \t" % (src, i, dist[i]),end=" ")
			self.printPath(parent,i)

	def dijkstra(self, src):
		row = len(self.graph)
		col = len(self.graph[0])
		dist = [1e+19] * row		# 输出数组。 dist[i] 将保存从 src 到 i 的最短距离 将所有距离初始化为 无穷大
		parent = [-1] * row				# 存储最短路径树的父数组
		dist[src] = 0					# 源顶点与自身的距离始终为 0
		queue = []						# 添加队列中的所有顶点
		for i in range(row):
			if sum(self.graph[i]) ==1:	## 当前节点与其他节点无交互关系，
				continue
			queue.append(i)			
		while queue:	# 查找所有顶点的最短路径
			u = self.minDistance(dist,queue)	# 从仍在队列queue中的顶点集中选取最小距离顶点
			queue.remove(u)	# remove min element	
			for i in range(col):	# 更新已选取顶点的相邻顶点的距离值和父索引。 只考虑那些仍在队列中的顶点
				# 仅当它在队列中时才更新 dist[i]，从 u 到 i 有一条边，并且从 src 通过 u 到 i 的路径的总权重小于 dist[i] 的当前值
				if self.graph[u][i] and i in queue:
					if dist[u] + self.graph[u][i] < dist[i]:
						dist[i] = dist[u] + self.graph[u][i]
						parent[i] = u
		# self.printSolution(src, dist,parent)
		all_sdp = []
		for j in range(len(self.graph)):
			sdp = self.get_sdp([],parent,j)
			all_sdp.append(sdp)
		return dist,all_sdp	## dist距离列表
