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
import pickle
import struct
from sklearn.cluster import DBSCAN,KMeans
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
parser.add_argument('--path', type=str,default='work',
                    help='')
parser.add_argument('--dataset', type=str,
                    help='')
parser.add_argument('--init', type=str, default='vec',
                    help='')
parser.add_argument('--mode', type=str, default = 'train',
                    help='')
parser.add_argument('--train_sample_per_epoch', type=str, default = 1000,
                    help='')
parser.add_argument('--regenerate_patch_pool',type=bool,default='False')

parser.add_argument('--write_patch_pool',type=bool,default='False')

parser.add_argument('--format',type=str, default='graphml',help='number of test sample size')

parser.add_argument('--DBSCAN_eps', type=float,default='4')

parser.add_argument('--DBSCAN_min_samples',type=int,default='5')

parser.add_argument('--sample_split_ratio',type=float,default=0.8)

parser.add_argument('--samples',type=float,default=1000)

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
	cluster_out_path='E:/cudaProjects/SurfPatchQt/cluster_data/'

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
	while num_of_files < start+num:
		file_path = './data/' + args.dataset + '/extracted_files'
		if os.path.exists(file_path+'/adjacent_matrix_'+'{:03d}'.format(num_of_files)+'.graphml'):
			g, features, adjacency_matrix = InitGraph(file_path, num_of_files)
			G.append(g)  # G is a dgl graph
			F.append(features)  # F is a n*m numpy array
			A.append(adjacency_matrix)  # A is a n*n numpy array
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
	#collect nodes and store them in patch_nodes
	while patch_nodes_queue and len(patch_nodes) < patch_size:

		node_id_temp=patch_nodes_queue.pop(0)
		if not g.has_nodes(node_id_temp):
			continue
		connected_node_pair=g.out_edges(node_id_temp,'uv')
		for connected_node in connected_node_pair[1].numpy():
			if connected_node not in patch_nodes and len(patch_nodes) <patch_size:
				patch_nodes.append(connected_node)
				patch_nodes_queue.append(connected_node)

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

	return ret_g, ret_adj_matrix, ret_node_feature,patch_nodes,

def testProcrustesDistance(G,F,A):
	g1,a1,f1,_=patchGenerator(G,F,A)
	Loss=ProcrustesLoss()
	p_distance=Loss.procrustes_distance(f1,f1)
	if(p_distance==0):
		print("correct")
	else:
		print(p_distance,"incorrect")

def train(GCN,G,F,A):
	total_sample_num = countFileNum("./data/" + args.dataset + "/surface_files")
	args.train_samples = args.sample_split_ratio * total_sample_num
	args.test_samples = (1.0 - args.sample_split_ratio) * total_sample_num

	random.seed()

	optimizer = optim.SGD(GCN.parameters(), lr=args.lr, momentum=0.9)

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
		if itera%100==0 or itera ==30 or itera==60 or itera==1:
			torch.save(GCN.state_dict(),path+args.dataset+'-'+'epochs-'+str(itera)+'-samples-'+str(args.train_samples)+'-init-'+args.init+'-GCN.pth')


def generatePatchPool():
#save graph file for a single file

	sample_num=countFileNum("./data/"+args.dataset+"/surface_files/")
	surface_idx=0
	patchG={}
	patchF={}
	patchA={}
	while(surface_idx<sample_num):
		surface_id=surface_idx
		format_surface_id = "{0:0=3d}".format(surface_id)  # to form idx to three digits
		surface_file_path = "./data/" + args.dataset + "/surface_files/" + args.dataset + "_surfaces_" + format_surface_id + ".bin"

		if (os.path.exists(surface_file_path)):
			print("Reading Surface:", surface_file_path)
			patchG[surface_id]=[]
			patchF[surface_id]=[]
			patchA[surface_id]=[]

			vertices, normals, indices = surfaceIO.ReadSurface(surface_file_path)
			vertices = np.asarray(vertices)
			numVertices=(int)(len(vertices)/3)
			nodes_covered = np.zeros((numVertices), dtype=bool)

			G, F, A = InitGraphs(surface_idx, 1)
			for node_id in range(0, numVertices):
				if not nodes_covered[node_id]:
					g, a, f, nodes_dict = patchGenerator(G,F,A,5,0,node_id)# g is a dgl graph

					patch_id=len(patchG)

					patchG[surface_id].append(g)
					patchA[surface_id].append(a)
					patchF[surface_id].append(f)

					for node in nodes_dict:
						nodes_covered[node]=True

		surface_idx+=1

	if args.write_patch_pool:

		patch_file_path = "./data/" + args.dataset + "/patch_pool/" + args.dataset + "_patch_pool"+".graphml"
		print("Writing patch:", patch_file_path)
		f=open(patch_file_path,'wb')
		pickle.dump(patchG)
		f.close()

		feature_file_path = "./data/" + args.dataset + "/patch_pool/" + args.dataset + "_patch_pool_feature"+ ".bin"
		f=open(feature_file_path,'wb')
		pickle.dump(patchF)
		f.close()
		print("Writing patch feature:", feature_file_path)

		adj_file_path = "./data/" + args.dataset + "/patch_pool/" + args.dataset + "_patch_pool_adj" + ".bin"
		f = open(adj_file_path, 'wb')
		pickle.dump(patchA)
		f.close()
		print("Writing patch feature:", adj_file_path)

	return patchG, patchF,patchA

def readPatchPool():

	surface_id=args.train_num
	#load dict of patch2SurfaceDict
	sample_num=countFileNum("./data/"+args.dataset+"/surface_files/")

	patch_file_path = "./data/" + args.dataset + "/patch_pool/" + args.dataset + "_patch_pool" + ".graphml"
	print("Writing patch:", patch_file_path)
	f = open(patch_file_path, 'rb')
	patchG=pickle.load(f)
	f.close()

	feature_file_path = "./data/" + args.dataset + "/patch_pool/" + args.dataset + "_patch_pool_feature" + ".bin"
	f = open(feature_file_path, 'rb')
	patchF=pickle.load(f)
	f.close()
	print("Writing patch feature:", feature_file_path)

	adj_file_path = "./data/" + args.dataset + "/patch_pool/" + args.dataset + "_patch_pool_adj" + ".bin"
	f = open(adj_file_path, 'rb')
	patchA=pickle.load(f)
	f.close()
	print("Writing patch feature:", adj_file_path)

	return patchG, patchF,patchA

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 40 epochs"""
    lr = args.lr * (0.1 ** (epoch // 40))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def evulate(itera):
	G = GCN()
	G.load_state_dict(torch.load(path+'/model/'+args.dataset+'-'+'epochs-'+str(itera)+'-samples-'+str(args.train_samples)+'-init-'+args.init+'-GCN.pth'))
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
		features.tofile(path+'Result/Result/'+args.dataset+'-node-features-'+'{:03d}'.format(id)+'-'+'epochs-'+str(itera)+'-samples-'+str(args.train_samples)+'-init-'+args.init+'.dat',format='<f')

def inference(itera):
	total_sample_num=countFileNum("./data/"+args.dataset+"/surface_files")

	args.train_samples=(int)((1.0-args.sample_split_ratio)*total_sample_num)
	args.test_samples=(int)(args.sample_split_ratio*total_sample_num)
	print(total_sample_num)
#	AllPatchG,AllPatchF,AllPatchA = generatePatchPool() if args.regenerate_patch_pool else readPatchPool()

	model = GCN()
	model.load_state_dict(torch.load(path+args.dataset+'-'+'epochs-'+str(itera)+'-samples-'+str(args.samples)+'-init-'+args.init+'-GCN.pth'))
	t = 0
	patch2SurfaceDict={}
	feature_space_matrix=[]
	for sid in range(args.train_samples,args.train_samples+args.test_samples):


		surface_id = sid
		format_surface_id = "{0:0=3d}".format(surface_id)  # to form idx to three digits
		surface_file_path = "./data/" + args.dataset + "/surface_files/" + args.dataset + "_surfaces_" + format_surface_id + ".bin"
		x = time.time()
		if (os.path.exists(surface_file_path)):

			vertices, normals, indices = surfaceIO.ReadSurface(surface_file_path)
			numVertices = (int)(len(vertices) / 3)
			nodes_covered = np.zeros((numVertices), dtype=bool)

			G, F, A = InitGraphs(surface_id, 1)

			if not G:
				continue

			for node_id in range(0, numVertices):
				if not nodes_covered[node_id]:
					g, a, f, nodes_dict = patchGenerator(G, F, A, 5, 0, node_id)  # g is a dgl graph
					f=torch.FloatTensor(f)
					a=torch.FloatTensor(a)

					with torch.no_grad():
						pred_feature = model(g, f)
					patch_id = len(feature_space_matrix)
					patch2SurfaceDict[patch_id] = sid
					feature_space_matrix.append(pred_feature.numpy().flatten())
					for node in nodes_dict:
						nodes_covered[node] = True

		y = time.time()
		print('Inference Time = ' + str(y - x))
		t += (y - x)

	feature_space_matrix=np.asarray(feature_space_matrix)
	labels=clustering(np.asarray(feature_space_matrix))

	#from patch label to surface id
	label2SurfaceId={}

	patch_id =0
	for label in labels:
		if label not in label2SurfaceId:
			label2SurfaceId[label] =[]
		if patch2SurfaceDict[patch_id] not in label2SurfaceId[label]:
			label2SurfaceId[label].append(patch2SurfaceDict[patch_id])
		patch_id+=1
	print(label2SurfaceId)
	with open(cluster_out_path+args.dataset+'_patch2Surface_dict'+'.bin', "wb") as f:
		numLabels=len(label2SurfaceId)
		print("numLabels:",numLabels)
		f.write(struct.pack('i', numLabels))#write label num
		for key,surfaceId_arr in label2SurfaceId.items():
			cluster_len=len(surfaceId_arr)
			f.write(struct.pack('i',cluster_len))#write how many object in the current cluster
			for sid in surfaceId_arr:
				f.write(struct.pack('i',sid))#each surface id in the current cluster

def clustering(X):
	#X should be a n*m numpy array where n is the object num and m is the feature num.
	clustering_model=DBSCAN(eps=args.DBSCAN_eps,min_samples=args.DBSCAN_min_samples)
	clustering_model.fit(X)
	print(clustering_model.labels_)
	return clustering_model.labels_

def countFileNum(dir,keyword=None):
	if not os.path.exists(dir):
		raise ValueError("%s does not exists.",dir)

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
	