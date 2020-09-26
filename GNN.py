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
from dgl.nn.pytorch import GraphConv,SAGEConv
from dgl import DGLGraph
import networkx as nx
import SurfaceProcesser as sp
import surfaceIO
import random
parser = argparse.ArgumentParser(description='PyTorch Implementation of V2V')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate of G')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training')
parser.add_argument('--epochs', type=int,default=100,
                    help='number of epochs to train (default: 500)')
parser.add_argument('--samples', type=int, default=1000,
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


args = parser.parse_args()
print(not args.no_cuda)
print(torch.cuda.is_available())
args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 30, 'pin_memory': True} if args.cuda else {}

Inference = {'benard':[5,25,50,1418,1495,1503,1575,1594,1657,1681,1888],'cylinder':[6,26,941,1060,1062,1424],
             'tornado':[4,13,53,103,1026,1047,1052,1169,1242,1329,1715,1784],'5cp':[8,33,84,440,1155,1167,1183,1310,1482,1706],
             'plume':[7,13,23,87,132,215,217,250,445]}
if args.path == 'crc':
	path = '/afs/crc.nd.edu/user/j/jhan5/GNN/'
else:
	path = '/data/junhan/GNN/'


def InitGraph(file_path,id):
	graph = nx.read_graphml(file_path+'/adjacent_matrix_'+'{:03d}'.format(id)+'.graphml')
	num_of_nodes = nx.number_of_nodes(graph)
	adjacency_matrix = nx.to_numpy_matrix(graph)
	adjacency_matrix = np.asarray(adjacency_matrix,dtype='<f')
	features=np.asarray([])
	if args.init == 'vec' or args.init == 'torsion' or args.init == 'curvature':
		features = np.fromfile(file_path+'/node_features_'+'{:03d}'.format(id)+'.bin',dtype='<f')
	features = features.reshape(5,num_of_nodes).transpose()
	if args.init == 'vec':
		features = features[:,0:3]
	elif args.init == 'torsion':
		features = features[:,3:4]
	elif args.init == 'curvature':
		features = features[:,4:]
	G = dgl.DGLGraph()
	G.add_nodes(num_of_nodes)
	for i in range(0,num_of_nodes):
		for j in range(0,i):
			if adjacency_matrix[i][j]!=0:
				G.add_edge(i,j)
	G.add_edges(G.nodes(), G.nodes())

	return G,features,adjacency_matrix


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

	def forward(self, g, features):
		x = self.ec1(g, features)
		x = self.ec2(g, x)
		x = self.ec3(g, x)
		f = self.ec4(g, x)
		return f


class ProcrustesLoss(nn.Module):
	def __init__(self):
		super(ProcrustesLoss, self).__init__()
		self.loss = nn.MSELoss()
	def forward(self,features):
		#todo define procrustes distance here
		m = 0

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

def patchGenerator(G,F,A):
	random.seed()

	size_graph=len(G)
	surface_id = random.randint(0,size_graph-1)

	g = G[surface_id] #graph
	a = A[surface_id] #adjacent matrix
	f = F[surface_id] #features

	feature_value_num=f.shape[1] # should be 3 if it's vec

	size_node=g.num_nodes()
	node_id=random.randint(0,size_node-1)
	print("size_node", size_node)
	print("node_id",node_id)
	patch_size=random.randrange(5,13,2)
	print("patch_size",patch_size)
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

	min_node_id=min(patch_nodes)

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

	return ret_g, ret_adj_matrix, ret_node_feature

def train(GCN,G,F,A):
	device = torch.device("cuda:0" if args.cuda else "cpu")
	optimizer = optim.Adam(GCN.parameters(), lr=args.lr,betas=(0.9,0.999))
	# if args.loss == 'shortest':
	# 	Loss = PairWiseLoss()
	# elif args.loss == 'adj':
	# 	Loss = MultiHopLoss()
	# elif args.loss == 'random_walk':
	# 	Loss = RandomWalkLoss()
	Loss=ProcrustesLoss()
	for itera in range(1,args.epochs+1):
		print("==========="+str(itera)+"===========")
		loss = 0
		x = time.time()
		for i in range(0,args.train_sample_per_epoch):
			patch_graph1,patch_adj_mat1,patch_feature1= patchGenerator(G,F,A)
			patch_graph2,patch_adj_mat2,patch_feature2= patchGenerator(G,F,A)
			patch_adj_mat1 = torch.FloatTensor(patch_adj_mat1)
			patch_feature1 = torch.FloatTensor(patch_feature1)
			patch_adj_mat2 = torch.FloatTensor(patch_adj_mat2)
			patch_feature2 = torch.FloatTensor(patch_feature2)

			if args.cuda:
				patch_adj_mat1=patch_adj_mat1.cuda()
				patch_feature1=patch_feature1.cuda()
				patch_adj_mat2=patch_adj_mat2.cuda()
				patch_feature2=patch_feature2.cuda()

			gcn_patch_feature1=GCN(patch_graph1,patch_feature1)
			gcn_patch_feature2=GCN(patch_graph2,patch_feature2)
			gcn_loss=Loss()

		# g = G[i]
			# f = F[i]
			# a = A[i]
			#
			# if args.cuda:
			# 	f = f.cuda()
			# 	a = a.cuda()
			#
			# node_features1 = GCN(g,f)
			#
			# gcn_loss=Loss()
			# loss += gcn_loss.item()
			# optimizer.zero_grad()
			# gcn_loss.backward()
			# optimizer.step()
		y = time.time()
		print("Time = "+str(y-x))
		print("Loss = "+str(loss))
		#if itera==40 or itera==80:
			#adjust_learning_rate(optimizer,itera)
		if itera%100==0 or itera ==30 or itera==60:
			torch.save(GCN.state_dict(),path+'/model/'+args.dataset+'-'+'epochs-'+str(itera)+'-samples-'+str(args.samples)+'-loss-'+args.loss+'-init-'+args.init+'-GCN.pth')


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
	G = GCN(6)
	G.load_state_dict(torch.load(path+'/model/'+args.dataset+'-'+'epochs-'+str(itera)+'-samples-'+str(args.samples)+'-loss-'+args.loss+'-init-'+args.init+'-GCN.pth'))
	t = 0
	file_path = path+'Data/'+args.dataset
	for id in range(1,2001):
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
		features.tofile(path+'fuse/'+args.dataset+'-node-features-'+'{:03d}'.format(id)+'-'+'epochs-'+str(itera)+'-samples-'+str(args.samples)+'-loss-'+args.loss+'-init-'+args.init+'.dat',format='<f')


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

	G = []
	F = []
	A = []
	num_of_files = 0
	idx = 0
	while idx<args.samples:
		file_path = './data/'+args.dataset+'/extracted_files'
		if os.path.exists(file_path):
			g, features, adjacency_matrix= InitGraph(file_path,num_of_files)
			G.append(g)# G is a dgl graph
			F.append(features) # F is a n*m numpy array
			A.append(adjacency_matrix)# A is a n*n numpy array
			print('Reading '+file_path)
			idx += 1
		num_of_files += 1

	p1,a1,f1=patchGenerator(G,F,A)
	print(p1)
	print(a1)
	print(f1)

	# model = GCN(6)
	# if args.cuda:
	# 	model.cuda()
	# train(model,G,F,A)
	# inference(args.epochs)
	# #evulate(100)

if __name__== "__main__":
	if args.mode =='train':
		main()
	else:
		#evulate(args.epochs)
		inference(args.epochs)
	