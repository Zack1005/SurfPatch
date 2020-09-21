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
parser.add_argument('--init', type=str, default='pos',
                    help='')
parser.add_argument('--loss', type=str, default = 'shortest',
                    help='')
parser.add_argument('--mode', type=str, default = 'train',
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
	elif args.init == 'pos+vec':
		features = features[:,0:6]
	shortest_path_length =  np.fromfile(file_path+'/shortest-path-'+'{:03d}'.format(id)+'.dat',dtype='int16')
	G = dgl.DGLGraph()
	G.add_nodes(num_of_nodes)
	for i in range(0,num_of_nodes):
		for j in range(0,i):
			if adjacency_matrix[i][j]!=0:
				G.add_edge(i,j)
	G.add_edges(G.nodes(), G.nodes())
	mask = np.zeros((num_of_nodes,num_of_nodes))
	paths = np.zeros((num_of_nodes,num_of_nodes))
	idx = 0
	for k in range(num_of_nodes):
		for l in range(k):
			mask[k][l] = 1
			paths[k][l] = shortest_path_length[idx]
			idx += 1
	return G,torch.FloatTensor(features),torch.FloatTensor(adjacency_matrix),torch.FloatTensor(mask),torch.FloatTensor(paths)


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
    def __init__(self,input_dim=3):
        super(GCN, self).__init__()
        self.ec1 = GraphConv(input_dim,64,activation=F.relu)
        self.ec2 = GraphConv(64,128,activation=F.relu)
        self.ec3 = GraphConv(128,128,activation=F.relu)
        self.ec4 = GraphConv(128,128,activation=F.relu)

    def forward(self, g, features):
    	x = self.ec1(g,features)
    	x = self.ec2(g,x)
    	x = self.ec3(g,x)
    	f = self.ec4(g,x)
    	return f

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

class RandomWalkLoss(object):
	def __init__(self, adj_lists, train_nodes):
		super(RandomWalkLoss, self).__init__()
		self.Q = 10
		self.N_WALKS = 5
		self.WALK_LEN = 1
		self.N_WALK_LEN = 5
		self.adj_lists = adj_lists
		self.train_nodes = train_nodes
		self.NEG = 10

	def random_walk(self,node):
		path = []
		pos = []
		if len(self.adj_lists[int(node)]) == 0:
			return path
		cur_pairs = []
		curr_node = node
		i = 1
		for i in range(self.N_WALKS):
			for j in range(self.WALK_LEN):
				neighs = self.adj_lists[int(curr_node)]
				next_node = random.choice(list(neighs))
				# self co-occurrences are useless
				if next_node != node and next_node in self.train_nodes:
					path.append(next_node)
					curr_node = next_node
		pos.extend([node,pos_node] for pos_node in path)
		return path

	def get_negtive_nodes(self, node):
		neg = []
		neighbors = set([node])
		frontier = set([node])
		for i in range(self.N_WALK_LEN):
			current = set()
			for outer in frontier:
				current |= set(self.adj_lists[int(outer)])
			rontier = current - neighbors
			neighbors |= current
		far_nodes = set(self.train_nodes) - neighbors
		neg_samples = random.sample(far_nodes, self.NEG) if self.NEG < len(far_nodes) else far_nodes
		neg.extend([(node, neg_node) for neg_node in neg_samples])
		return neg_samples

	def random_walk_loss(self,features):
		loss = 0
		similarity = torch.mm(features,torch.transpose(features,0,1))
		#similarity = torch.exp(similarity)
		node_score = []
		for node in self.train_nodes:
			pos = self.random_walk(node)
			
			neg = self.get_negtive_nodes(node)
			#indexs = [list(x) for x in zip(*neg)]
			#node_indexs = [x for x in indexs[0]]
			#neighb_indexs = [x for x in indexs[1]]
			#neg_score = F.cosine_similarity(features[node_indexs],features[neighb_indexs])
			neg_score = similarity[node][neg]
			neg_score = self.Q*torch.mean(torch.log(torch.sigmoid(-neg_score)), 0)

			# multiple positive score
			#indexs = [list(x) for x in zip(*pos)]
			#node_indexs = [x for x in indexs[0]]
			#neighb_indexs = [x for x in indexs[1]]
			#pos_score = F.cosine_similarity(features[node_indexs],features[neighb_indexs])
			pos_score = similarity[node][pos]
			pos_score = torch.log(torch.sigmoid(pos_score))
			node_score.append(torch.mean(-pos_score-neg_score).view(1,-1))
				
		loss = torch.mean(torch.cat(node_score, 0))
	
		return loss


def train(GCN,G,F,A,P,M):
	device = torch.device("cuda:0" if args.cuda else "cpu")
	optimizer = optim.Adam(GCN.parameters(), lr=args.lr,betas=(0.9,0.999))
	if args.loss == 'shortest':
		Loss = PairWiseLoss()
	elif args.loss == 'adj': 
		Loss = MultiHopLoss()
	elif args.loss == 'random_walk':
		Loss = RandomWalkLoss()
	for itera in range(1,args.epochs+1):
		print("==========="+str(itera)+"===========")
		loss = 0
		x = time.time()
		for i in range(0,len(G)):
			g = G[i]
			f = F[i]
			a = A[i]
			p = P[i]
			m = M[i]
			
			if args.cuda:
				f = f.cuda()
				a = a.cuda()
				p = p.cuda()
				m = m.cuda()
	
			node_features = GCN(g,f)
			if args.loss == 'shortest':
				gcn_loss = Loss(node_features,p,m)
			elif args.loss == 'adj':
				gcn_loss = Loss(node_features,a,m)
			loss += gcn_loss.item()
			optimizer.zero_grad()
			gcn_loss.backward()
			optimizer.step()
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
	P = []
	M = []
	num_of_files = 1
	idx = 1
	while idx<=args.samples:
		file_path = path+'Data/'+args.dataset
		if os.path.exists(file_path+'/adjacency-matrix-'+'{:03d}'.format(num_of_files)+'.graphml'):
			g, features, adjacency_matrix, mask,paths = InitGraph(file_path,num_of_files)
			G.append(g)
			F.append(features)
			A.append(adjacency_matrix)
			P.append(paths)
			M.append(mask)
			print('Reading '+str(num_of_files)+'th graph')
			idx += 1
		num_of_files += 1
	model = GCN(6)
	if args.cuda:
		model.cuda()
	train(model,G,F,A,P,M)
	inference(args.epochs)
	#evulate(100)

if __name__== "__main__":
	if args.mode =='train':
		main()
	else:
		#evulate(args.epochs)
		inference(args.epochs)
	