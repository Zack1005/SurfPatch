import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import argparse
import copy
import torch.optim as optim

import sys
from torch.nn import init
import dgl
from dgl.nn.pytorch import GraphConv,SAGEConv, SumPooling
from dgl import DGLGraph
import networkx as nx
import SurfaceProcesser as sp
import surfaceIO
import random
import re

parser = argparse.ArgumentParser(description='PyTorch Implementation of V2V')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate of G')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training')
parser.add_argument('--epochs', type=int,default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--train_samples', type=int, default=1000,
                    help='number of training samples')
parser.add_argument('--path', type=str,default='work',
                    help='')
parser.add_argument('--dataset', type=str,
                    help='')
parser.add_argument('--init', type=str, default='vec',
                    help='')
parser.add_argument('--loss', type=str, default = 'shortest',
                    help='')
parser.add_argument('--mode', type=str, default = 'train',
                    help='')
parser.add_argument('--train_sample_per_epoch', type=str, default = 1000,
                    help='')
parser.add_argument('--test_samples',type=int, default=800,help='number of test sample size')

parser.add_argument('--regenerate_patch_pool',type=bool,default='False')

parser.add_argument('--write_patch_pool',type=bool,default='False')

parser.add_argument('--format',type=str, default='graphml',help='number of test sample size')



args = parser.parse_args()
print("Enable cuda?", not args.no_cuda)
print("Cuda device available?", torch.cuda.is_available())
args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 30, 'pin_memory': True} if args.cuda else {}
device = torch.device("cuda:0" if args.cuda else "cpu")
print("device:",device)
Inference = {'benard':[5,25,50,1418,1495,1503,1575,1594,1657,1681,1888],'cylinder':[6,26,941,1060,1062,1424],
             'tornado':[4,13,53,103,1026,1047,1052,1169,1242,1329,1715,1784],'5cp':[8,33,84,440,1155,1167,1183,1310,1482,1706],
             'plume':[7,13,23,87,132,215,217,250,445]}
if args.path == 'crc':
	path = '/afs/crc.nd.edu/user/j/jhan5/GNN/'
else:
	path = './data/model_snapshot/'


def InitGraph(file_path,id):
	graph = nx.read_graphml(file_path+'/adjacent_matrix_'+'{:03d}'.format(id)+'.graphml')
	print('Reading '+file_path+'/adjacent_matrix_'+'{:03d}'.format(id)+'.graphml')
	num_of_nodes = nx.number_of_nodes(graph)
	adjacency_matrix = nx.to_numpy_matrix(graph)
	adjacency_matrix = np.asarray(adjacency_matrix,dtype='<f')

	features = np.fromfile(file_path+'/node_features_'+'{:03d}'.format(id)+'.bin',dtype='<f')
	print('Reading ' + file_path+'/node_features_'+'{:03d}'.format(id)+'.bin')
	features = features.reshape(5,num_of_nodes).transpose()
	if args.init == 'vec':
		features = features[:,0:3]
	elif args.init == 'torsion':
		features = features[:,3:4]
	elif args.init == 'curvature':
		features = features[:,4:]
	G = dgl.DGLGraph()
	G=dgl.from_networkx(graph)
	# G.add_nodes(num_of_nodes)
	# for i in range(0,num_of_nodes):
	# 	for j in range(0,i):
	# 		if adjacency_matrix[i][j]!=0:
	# 			G.add_edge(i,j)
	# G.add_edges(G.nodes(), G.nodes())

	return G,features,adjacency_matrix

def InitGraphs(start, num):
	G = []
	F = []
	A = []
	num_of_files = start
	idx = start
	while idx < start+num:
		file_path = './data/' + args.dataset + '/extracted_files'
		if os.path.exists(file_path):
			g, features, adjacency_matrix = InitGraph(file_path, num_of_files)
			G.append(g)  # G is a dgl graph
			F.append(features)  # F is a n*m numpy array
			A.append(adjacency_matrix)  # A is a n*n numpy array
			idx += 1
		num_of_files += 1
	return G, F, A

def GenerateAdjList(adjacency_matrix):
	adj_list = []
	for i in range(0,len(adjacency_matrix)):
		nei_nodes = []
		for j in range(0,len(adjacency_matrix)):
			if j!=i and adjacency_matrix[i][j]==1:
				nei_nodes.append(j)
		adj_list.append(nei_nodes)
	return adj_list


def GetKOrderAdjacencyMatrix(adjacency_matrix,k):
	a = np.copy(adjacency_matrix)
	for i in range(1,k):
		a = np.matmul(a,adjacency_matrix.transpose())
	return a 


class GCN(nn.Module):
	def __init__(self, input_dim=3):
		super(GCN,self).__init__()
		self.ec1 = GraphConv(input_dim, 64, activation=F.relu)
		self.ec2 = GraphConv(64, 128, activation=F.relu)
		self.ec3 = GraphConv(128, 128, activation=F.relu)
		self.ec4 = GraphConv(128, 128, activation=F.relu)
		self.pooling = SumPooling()
		self.MLP=nn.Linear(128,128)

	def forward(self, g, features):
		x = self.ec1(g, features)
		x = self.ec2(g, x)
		x = self.ec3(g, x)
		x = self.ec4(g, x)
		x = self.pooling(g,x)
		f = self.MLP(x)
		return f


class ProcrustesLoss(nn.Module):
	def __init__(self):
		super(ProcrustesLoss, self).__init__()
		self.loss=nn.L1Loss()

	def orthogonal_procrustes(self,A, B, check_finite=True):
		# if check_finite:
		# 	np.asarray_chkfinite(A.numpy())
		# 	np.asarray_chkfinite(B.numpy())
		# else:
		# 	A = np.asanyarray(A)
		# 	B = np.asanyarray(B)
		if A.ndim != 2:
			raise ValueError('expected ndim to be 2, but observed %s' % A.ndim)
		if A.shape != B.shape:
			raise ValueError('the shapes of A and B differ (%s vs %s)' % (
				A.shape, B.shape))
		# Be clever with transposes, with the intention to save memory.
		u, w, vt = torch.svd(B.T.mm(A).T)
		R = u.mm(vt)
		scale = w.sum()
		return R, scale

	def procrustes_distance(self,patch_feature1,patch_feature2):
		mtx1 = patch_feature1
		mtx2 = patch_feature2

		if mtx1.ndim != 2 or mtx2.ndim != 2:
			raise ValueError("Input matrices must be two-dimensional")
		if mtx1.shape != mtx2.shape:
			print(mtx1.shape,mtx2.shape)
			raise ValueError("Input matrices must be of same shape")
		if mtx1.size == 0:
			raise ValueError("Input matrices must be >0 rows and >0 cols")

		# translate all the data to the origin
		mtx1 -= torch.mean(mtx1, 0)
		mtx2 -= torch.mean(mtx2, 0)

		norm1 = torch.norm(mtx1)
		norm2 = torch.norm(mtx2)

		if norm1 == 0 or norm2 == 0:
			raise ValueError("Input matrices must contain >1 unique points")

		# change scaling of data (in rows) such that trace(mtx*mtx') = 1
		mtx1 /= norm1
		mtx2 /= norm2

		# transform mtx2 to minimize disparity
		R, s = self.orthogonal_procrustes(mtx1, mtx2)
		mtx2 = torch.mm(mtx2, R.T) * s

		# measure the dissimilarity between the two datasets
		disparity = torch.sqrt(torch.sum(torch.square(mtx1 - mtx2)))
		return disparity

	def forward(self,patch_feature1,patch_feature2, gcn_feature1,gcn_feature2):
		p_distance=self.procrustes_distance(patch_feature1,patch_feature2)
		feature_distance_of_gcn=(torch.norm(gcn_feature1-gcn_feature2,p=2))

		return self.loss(feature_distance_of_gcn,p_distance)

class PairWiseLoss(nn.Module):
	def __init__(self):
		super(PairWiseLoss,self).__init__()
		self.loss = nn.MSELoss()

	def forward(self,features,shortest_path_length,mask):
		n = features.size()[0]
		norm = torch.sum(features,dim=1,keepdim=True)
		norm = norm.expand(n,n)
		distance = norm+norm.t()-2*features.mm(features.t())
		return self.loss(distance*mask,shortest_path_length)


class MultiHopLoss(nn.Module):
	def __init__(self):
		super(MultiHopLoss,self).__init__()

	def forward(self,features,adjacency_matrix,mask):
		similarity = torch.mm(features,torch.transpose(features,0,1)) 
		return nn.MSELoss()(similarity*mask,adjacency_matrix*mask)

def patchGenerator(G,F,A,patch_size,surface_id=None,node_id=None):

	if surface_id is None:
		size_graph=len(G)
		surface_id = random.randint(0,size_graph-1)

	g = G[surface_id] #graph
	a = A[surface_id] #adjacent matrix
	f = F[surface_id] #features

	feature_value_num=f.shape[1] # should be 3 if it's vec

	if node_id is None:
		size_node=g.num_nodes()
		node_id=random.randint(0,size_node-1)

	ret_g=dgl.DGLGraph()

	ret_adj_matrix=np.zeros([patch_size,patch_size])
	ret_node_feature=np.zeros([patch_size,feature_value_num])

	patch_nodes_queue = [node_id]  # a queue structure
	patch_nodes=[node_id]
	cnt=1
	#collect nodes and store them in patch_nodes
	while patch_nodes_queue and cnt < patch_size:

		node_id_temp=patch_nodes_queue.pop(0)
		connected_node_pair=g.out_edges(node_id_temp,'uv')
		for connected_node in connected_node_pair[1].numpy():
			if(connected_node not in patch_nodes):
				patch_nodes.append(connected_node)
				patch_nodes_queue.append(connected_node)
				cnt+=1
				if cnt>=patch_size:
					break

	#init ret_g and ret_adj_matrix
	ret_g.add_nodes(len(patch_nodes))

	for i in range(0,len(patch_nodes)):
		for j in range(i,len(patch_nodes)):
			node1=patch_nodes[i]
			node2=patch_nodes[j]
			if(node1!=node2 and a[node1][node2]!=0):
				ret_adj_matrix[i][j]=1
				ret_adj_matrix[j][i]=1
				ret_g.add_edge(i,j)

	#init ret_node_feature
	for i in range(0,len(patch_nodes)):
		ret_node_feature[i]=f[patch_nodes[i]]

	ret_g=dgl.add_self_loop(ret_g)

	return ret_g, ret_adj_matrix, ret_node_feature,patch_nodes

def testProcrustesDistance(G,F,A):
	g1,a1,f1=patchGenerator(G,F,A)
	Loss=ProcrustesLoss()
	p_distance=Loss.procrustes_distance(f1,f1)
	if(p_distance==0):
		print("correct")
	else:
		print(p_distance,"incorrect")

def train(GCN,G,F,A):
	random.seed()

	optimizer = optim.SGD(GCN.parameters(), lr=torch.tensor(args.lr,device=device), momentum=torch.tensor(0.9,device=device))

	Loss=ProcrustesLoss()

	for itera in range(1,args.epochs+1):
		print("==========="+str(itera)+"===========")
		loss = torch.tensor([0.0],device=device)
		x = time.time()
		for i in range(0,args.train_sample_per_epoch):
			patch_size=5

			patch_graph1,patch_adj_mat1,patch_feature1,_= patchGenerator(G,F,A,patch_size)
			while (patch_graph1.num_nodes() is not patch_size):
				patch_graph1, patch_adj_mat1, patch_feature1, _ = patchGenerator(G, F, A, patch_size)
			patch_graph2,patch_adj_mat2,patch_feature2,_= patchGenerator(G,F,A,patch_size)
			while(patch_graph2.num_nodes() is not patch_size):
				patch_graph2, patch_adj_mat2, patch_feature2, _ = patchGenerator(G, F, A, patch_size)

			patch_adj_mat1 = torch.FloatTensor(patch_adj_mat1)
			patch_feature1 = torch.FloatTensor(patch_feature1)
			patch_adj_mat2 = torch.FloatTensor(patch_adj_mat2)
			patch_feature2 = torch.FloatTensor(patch_feature2)

			if args.cuda:
				patch_graph1 = patch_graph1.to(device)
				patch_adj_mat1=patch_adj_mat1.to(device)
				patch_feature1=patch_feature1.to(device)
				patch_graph2 = patch_graph2.to(device)
				patch_adj_mat2=patch_adj_mat2.to(device)
				patch_feature2=patch_feature2.to(device)

			gcn_patch_feature1=GCN(patch_graph1,patch_feature1)
			gcn_patch_feature2=GCN(patch_graph2,patch_feature2)

			gcn_loss=Loss(patch_feature1,patch_feature2,gcn_patch_feature1,gcn_patch_feature2).to(device)
			loss+=gcn_loss.item()
			optimizer.zero_grad()
			gcn_loss.backward()
			optimizer.step()

		y = time.time()
		print("Time = "+str(y-x))
		print("Loss = "+str(loss))
		#if itera==40 or itera==80:
			#adjust_learning_rate(optimizer,itera)
		if itera%100==0 or itera ==30 or itera==60:
			torch.save(GCN.state_dict(),path+args.dataset+'-'+'epochs-'+str(itera)+'-samples-'+str(args.samples)+'-loss-'+args.loss+'-init-'+args.init+'-GCN.pth')


def generatePatchPool():
#save graph file for a single file
	G,F,A=InitGraphs(args.train_samples,args.test_samples)

	surface_id=0
	patchG=[]
	patchF=[]
	patchA=[]
	surfaceId_array=[]
	nodes_dict_array=[]
	while(surface_id<args.test_samples):

		caseId = "{0:0=3d}".format(surface_id+args.train_samples)  # to form idx to three digits
		surface_file_path = "./data/" + args.dataset + "/surface_files/" + args.dataset + "_surfaces_" + caseId + ".bin"
		if (os.path.exists(surface_file_path)):
			vertices, normals, indices = surfaceIO.ReadSurface(surface_file_path)
			vertices = np.asarray(vertices)
			numVertices=(int)(len(vertices)/3)
			nodes_covered = np.zeros((numVertices), dtype=bool)
			patch_idx=0
			for node_id in range(0, numVertices):
				if not nodes_covered[node_id]:
					patch_id="{0:0=3d}".format(patch_idx)
					g, a, f, nodes_dict = patchGenerator(G,F,A,surface_id,node_id)# g is a dgl graph
					g= dgl.to_networkx(g)
					patchG.append(g)
					patchA.append(a)
					patchF.append(f)
					surfaceId_array.append(surface_id)
					nodes_dict_array.append(nodes_dict)
					nodes_dict=np.asarray(nodes_dict)
					if args.write_patch_pool:
						patch_file_path="./data/"+args.dataset+"/patch_pool/"+args.dataset+"_surface_"+caseId+"_patch_"+patch_id+".graphml"
						node_dict_file_path="./data/"+args.dataset+"/patch_pool/"+args.dataset+"_surface_"+caseId+"_node_dict_"+patch_id+".bin"
						nx.write_graphml(g, patch_file_path)
						nodes_dict.tofile(node_dict_file_path,format='<f')

					for node in nodes_dict:
						nodes_covered[node]=True
					patch_idx+=1

		surface_id+=1
	return patchG, patchF,patchA,nodes_dict_array,surfaceId_array

def readPatchPool():


	surface_id=args.train_num
	patchG=[]
	patchF=[]
	patchA=[]
	surfaceId_array=[]
	nodes_dict_array=[]
	while(surface_id<args.train_num+args.test_num):
		caseId = "{0:0=3d}".format(surface_id)  # to form idx to three digits
		patch_num=countFileNum("./data/" + args.dataset + "/patch_pool/",keyword="_surface_" + caseId + "_patch_+")

		surface_graph_file_path = './data/' + args.dataset + '/extracted_files'

		G, F, A = InitGraph(surface_graph_file_path,surface_id)

		for patch_idx in range(0,patch_num):
			patch_id="{0:0=3d}".format(patch_idx)
			patch_file_path = "./data/" + args.dataset + "/patch_pool/" + args.dataset + "_surface_" + caseId + "_patch_" + patch_id + ".graphml"
			node_dict_file_path = "./data/" + args.dataset + "/patch_pool/" + args.dataset + "_surface_" + caseId + "_node_dict_" + patch_id + ".bin"

			g=nx.read_graphml(patch_file_path)
			num_of_nodes_per_patch=nx.number_of_nodes(g)
			patchG.append(g)

			nodes_dict=np.fromfile(node_dict_file_path)
			surfaceId_array.append(surfaceId_array)
			nodes_dict_array.append(nodes_dict)

			patch_feature=[]
			for node in nodes_dict:
				patch_feature.append(F[node])
			patchF.append(patch_feature)

			patch_adj = nx.to_numpy_matrix(g)
			patch_adj = np.asarray(patch_adj, dtype='<f')
			patchA.append(patch_adj)

	return patchG, patchF,patchA,nodes_dict_array,surfaceId_array

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 40 epochs"""
    lr = args.lr * (0.1 ** (epoch // 40))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def evulate(itera):
	G = GCN()
	G.load_state_dict(torch.load(path+'/model/'+args.dataset+'-'+'epochs-'+str(itera)+'-samples-'+str(args.samples)+'-loss-'+args.loss+'-init-'+args.init+'-GCN.pth'))
	t = 0
	file_path = path+'Data/'+args.dataset
	for id in Inference[args.dataset]:
		graph = nx.read_graphml(file_path+'/adjacency-matrix-'+'{:03d}'.format(id)+'.graphml')
		num_of_nodes = nx.number_of_nodes(graph)
		adjacency_matrix = nx.to_numpy_matrix(graph)
		adjacency_matrix = np.asarray(adjacency_matrix,dtype='<f')
		if args.init == 'pos' or args.init == 'vec' or args.init == 'pos+vec':
			features = np.fromfile(file_path+'/nodes-features-'+'{:03d}'.format(id)+'.dat',dtype='<f')
		elif args.init == 'normal':
			features = np.fromfile(file_path+'/nodes-features-normals-'+'{:03d}'.format(id)+'.dat',dtype='<f')
		features = features.reshape(6,num_of_nodes).transpose()
		if args.init == 'pos':
			features = features[:,0:3]
		elif args.init == 'vec' or args.init == 'normal':
			features = features[:,3:6]
		g = dgl.DGLGraph()
		g.add_nodes(num_of_nodes)
		for i in range(0,num_of_nodes):
			for j in range(0,i):
				if adjacency_matrix[i][j]!=0:
					g.add_edge(i,j)
		g.add_edges(g.nodes(), g.nodes())
		features = torch.FloatTensor(features)
		x = time.time()
		with torch.no_grad():
			node_features = G(g,features)
		y = time.time()
		print('Inference Time = '+str(y-x))
		t += (y-x)
		features = node_features.numpy()
		features = np.asarray(features,dtype='<f')
		features = features.flatten('F')
		features.tofile(path+'Result/Result/'+args.dataset+'-node-features-'+'{:03d}'.format(id)+'-'+'epochs-'+str(itera)+'-samples-'+str(args.samples)+'-loss-'+args.loss+'-init-'+args.init+'.dat',format='<f')

def inference(itera):

	if(args.regenerate_patch_pool):
		generatePatchPool()

	# G = GCN()
	# G.load_state_dict(torch.load(path+args.dataset+'-'+'epochs-'+str(itera)+'-samples-'+str(args.samples)+'-loss-'+args.loss+'-init-'+args.init+'-GCN.pth'))
	# t = 0
	# file_path = path+'Data/'+args.dataset
	# for id in range(3000,4000):
	# 	graph = nx.read_graphml(file_path+'/adjacenct_matrix_'+'{:03d}'.format(id)+'.graphml')
	# 	num_of_nodes = nx.number_of_nodes(graph)
	# 	adjacency_matrix = nx.to_numpy_matrix(graph)
	# 	adjacency_matrix = np.asarray(adjacency_matrix,dtype='<f')
	# 	features = np.fromfile(file_path+'/node_features_'+'{:03d}'.format(id)+'.bin',dtype='<f')
	# 	features = features.reshape(5,num_of_nodes).transpose()
	# 	if args.init == 'vec':
	# 		features = features[:,0:3]
	# 	elif args.init == 'torsion':
	# 		features = features[:,3:4]
	# 	elif args.init=='curvature':
	# 		features = features[:,4:]
	# 	g = dgl.DGLGraph()
	# 	g.add_nodes(num_of_nodes)
	# 	for i in range(0,num_of_nodes):
	# 		for j in range(0,i):
	# 			if adjacency_matrix[i][j]!=0:
	# 				g.add_edge(i,j)
	# 	g.add_edges(g.nodes(), g.nodes())
	# 	features = torch.FloatTensor(features)
	# 	x = time.time()
	# 	with torch.no_grad():
	# 		node_features = G(g,features)
	# 	y = time.time()
	# 	print('Inference Time = '+str(y-x))
	# 	t += (y-x)
	# 	features = node_features.numpy()
	# 	features = np.asarray(features,dtype='<f')
	# 	features = features.flatten('F')
	# 	features.tofile(path+'fuse/'+args.dataset+'-node-features-'+'{:03d}'.format(id)+'-'+'epochs-'+str(itera)+'-samples-'+str(args.samples)+'-loss-'+args.loss+'-init-'+args.init+'.bin',format='<f')

def countFileNum(dir,keyword=None):

	if keyword is None:
		return len(os.listdir(dir))
	count=0
	for path in os.listdir(dir):
		if keyword in path:
			count+=1
	return count

def weights_init_kaiming(m):
	classname = m.__class__.__name__
	if classname.find("Conv")!=-1:
		init.kaiming_uniform_(m.weight.data)
	elif classname.find("Linear")!=-1:
		init.kaiming_uniform_(m.weight.data)
	elif classname.find("BatchNorm")!=-1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)

def main():

	G,F,A=InitGraphs(0,args.train_samples)

#	testProcrustesDistance(G,F,A)

	model = GCN()
	if args.cuda:
		model.cuda(device)
	print(model)
	train(model,G,F,A)
#	inference(args.epochs)
	# #evulate(100)

if __name__== "__main__":
	if args.mode =='train':
		main()
	else:
		#evulate(args.epochs)
		inference(args.epochs)
	