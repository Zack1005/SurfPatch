import numpy as np
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.cluster import DBSCAN, KMeans
import scipy as sp
import pickle
import os
import struct
import networkx as nx
color_map = ["#4DA9FF", # light blue
             "#CC99FF", # light purple
             "#FF3333", # light red
             "#FFFF00", # light yellow
             "#66FF66", # light green
             "#CC9966", # light brown
             "#336600", # dark green
             "#FF99CC", # light pink
             "#6666FF", # violet
             "#FF9900", # orange
             "#8EE5EE",
             "#EE1289",
             "#FFFAEB9"
            ]


Inference = {'benard':[5,25,50,1418,1495,1503,1575,1594,1657,1681,1888],'cylinder':[6,26,941,1060,1062,1424],
             'tornado':[4,13,53,103,1026,1047,1052,1169,1242,1329,1715,1784],'5cp':[8,33,84,440,1155,1167,1183,1310,1482,1706],
             'plume':[7,13,23,87,132,215,217,250,445],'swirls':[1802,1790,1765,1696,1690,1632,13]}

def vecMult(v1,v2):
    if not isinstance(v1,list) or not isinstance(v2,list):
        return
    return [a*b for a, b in zip(v1,v2)]

def DrawScatter(graph,label=None):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	x = []
	y = []
	z = []
	for i in range(0,len(graph)):
		x.append(graph[i][0])
		y.append(graph[i][1])
		z.append(graph[i][2])
	ax.scatter(x,y,z,c=label)
	ax.axis('off')
	plt.show()


def ComputeCurvature(v1,v2):
	cosv = np.sum(v1*v2)/np.sqrt((np.sum(v1*v1)*np.sum(v2*v2)))
	cosv = np.clip(cosv,-1,1)
	return np.arccos(cosv)

def cross(v1,v2):
	v = np.zeros((3))
	v[0] = v1[1]*v2[2]-v1[2]*v2[1]
	v[1] = v1[2]*v2[0]-v1[0]*v2[2]
	v[2] = v1[0]*v2[1]-v1[1]*v2[0]
	return v

def ComputeTorsion(v1,v2,v3):
	len1 = np.sqrt(np.sum(v1*v1))
	len2 = np.sqrt(np.sum(v2*v2))
	len3 = np.sqrt(np.sum(v3*v3))
	tmp1 = np.sum(v1*v2)/(len1*len2)
	tmp2 = np.sum(v2*v3)/(len2*len3)
	if (tmp1>0.99 or tmp1<-0.99 or tmp2>0.99 or tmp2<-0.99):
		return 0
	else:
		cosv = (tmp1*tmp2-(np.sum(v1*v3))/(len1*len3))/(np.sqrt(1-tmp1*tmp1)*np.sqrt(1-tmp2*tmp2))
		cosv = np.clip(cosv,-1,1)
		if np.sum(cross(v1,v2)*v3)<0:
			return -np.arccos(cosv)
		else:
			return np.arccos(cosv)

def GetVelocity(vec,pos,dim_x,dim_y,dim_z):
	if(pos[0]<=0.0000001 or pos[0]>=dim_x-1.0000001 or pos[1]<=0.0000001 or pos[1]>=dim_y-1.0000001 or pos[2]<=0.0000001 or pos[2]>=dim_z-1.0000001):
		return False,None

	x = int(pos[0])
	y = int(pos[1])
	z = int(pos[2])

	vecf1 = np.array([vec[0][x][y][z],vec[1][x][y][z],vec[2][x][y][z]])
	vecf2 = np.array([vec[0][x+1][y][z],vec[1][x+1][y][z],vec[2][x+1][y][z]])
	vecf3 = np.array([vec[0][x][y+1][z],vec[1][x][y+1][z],vec[2][x][y+1][z]])
	vecf4 = np.array([vec[0][x+1][y+1][z],vec[1][x+1][y+1][z],vec[2][x+1][y+1][z]])
	vecf5 = np.array([vec[0][x][y][z+1],vec[1][x][y][z+1],vec[2][x][y][z+1]])
	vecf6 = np.array([vec[0][x+1][y][z+1],vec[1][x+1][y][z+1],vec[2][x+1][y][z+1]])
	vecf7 = np.array([vec[0][x][y+1][z+1],vec[1][x][y+1][z+1],vec[2][x][y+1][z+1]])
	vecf8 = np.array([vec[0][x+1][y+1][z+1],vec[1][x+1][y+1][z+1],vec[2][x+1][y+1][z+1]])

	facx = pos[0]-x
	facy = pos[1]-y
	facz = pos[2]-z

	ret = (1-facx)*(1-facy)*(1-facz)*vecf1+(facx)*(1-facy)*(1-facz)*vecf2+(1-facx)*(facy)*(1-facz)*vecf3+(facx)*(facy)*(1-facz)*vecf4+(1-facx)*(1-facy)*(facz)*vecf5+(facx)*(1-facy)*(facz)*vecf6+(1-facx)*(facy)*(facz)*vecf7+(facx)*(facy)*(facz)*vecf8
	return True,ret



def WriteGraph(vertices,indices,vec,dim_x,dim_y,dim_z,matrix_file=None,node_feature_file=None):
	print(matrix_file)
	print(node_feature_file)
	G = nx.Graph()
	for i in range(0,len(vertices),3):
		G.add_node((int)(i/3))
	for j in range(0,len(indices),4):
		G.add_edge(indices[j+0],indices[j+1])
		G.add_edge(indices[j+2],indices[j+1])
		G.add_edge(indices[j+0],indices[j+2])
		G.add_edge(indices[j+2],indices[j+3])
		G.add_edge(indices[j+1],indices[j+3])
	print(nx.is_connected(G))
	if nx.is_connected(G):
		#adjacency_matrix = nx.to_numpy_matrix(G)
		#adjacency_matrix = np.asarray(adjacency_matrix,dtype='int16')
		#adjacency_matrix = adjacency_matrix.flatten('F')
		#adjacency_matrix.tofile(matriwrite_graphml_lxmlx_file,format='int16')
		nx.write_graphml(G,matrix_file)

		nodes = list(G.nodes)
		#print(mapping)
		node_features = np.zeros((len(nodes),5))
		for i in range(0,len(nodes)):
			pos =np.asarray([vertices[nodes[i]*3],vertices[nodes[i]*3+1],vertices[nodes[i]*3+2]])

			#compute velocity of vertex at i
			b, v = GetVelocity(vec, pos, dim_x, dim_y, dim_z)
			if b:
				node_features[i][0:3] = v
				pos_prev = pos - v * 0.001
				pos_follow = pos + v * 0.001

				b_prev, v_prev = GetVelocity(vec, pos_prev, dim_x, dim_y, dim_z)
				b_follow, v_follow = GetVelocity(vec, pos_follow, dim_x, dim_y, dim_z)

				# compute torsion
				torsion = 0
				if b_prev and b_follow:
					torsion = ComputeTorsion(pos_prev, pos, pos_follow)
				node_features[i][3:4] = torsion

				# compute curvature
				curvature = 0
				if b_follow:
					curvature = ComputeCurvature(pos, pos_follow)

				node_features[i][4:] = curvature
			else :
				node_features[i][0:3] = 0
				node_features[i][3:4] = 0
				node_features[i][4:] = 0

		node_features = np.asarray(node_features,dtype='<f')
		node_features = node_features.flatten('F')
		node_features.tofile(node_feature_file,format='<f')
		# if idx<=1500:
		# 	short_path = []
		# 	paths = dict(nx.shortest_path_length(G))
		# 	'''
		# 	output = open(shortest_path_file,'wb')
		# 	output.dump(paths,output)
		# 	output.close()
		# 	'''
		# 	for i in list(G.nodes):
		# 		for k in range(0,i):
		# 			short_path.append(paths[i][k])
		# 	short_path = np.asarray(short_path,dtype='int16')
		# 	#print(values-short_path)
		# 	short_path = short_path.flatten('F')
		# 	short_path.tofile(shortest_path_file,format='int16')


def ReadVec(file,dim_x,dim_y,dim_z):
	v = np.fromfile(file,dtype='<f')
	v = v.reshape(dim_z,dim_y,dim_x,3).transpose()
	return v

def getcolor(labels):
	colors = []
	for i in range(0,len(labels)):
		colors.append(color_map[labels[i]])
	return colors

def getVertices(file):
	data = open(file,'rb')
	s = str(data.readline(),'utf-8').strip()
	point = s.split('\t')
	vertices = np.zeros((len(point)//3,3))
	for i in range(0,len(point),3):
		vertices[i//3][0] = float(point[i])
		vertices[i//3][1] = float(point[i+1])
		vertices[i//3][2] = float(point[i+2])
	return vertices

def TSNE(id,dataset,itera,samples,loss,init):
	if os.path.exists('/Volumes/Research/GNN/Flow-Surface/Result/'+dataset+'-node-features-'+'{:03d}'.format(id)+'-epochs-'+str(itera)+'-samples-'+str(samples)+'-loss-'+str(loss)+'-init-'+str(init)+'.dat'):
		features = np.fromfile('/Volumes/Research/GNN/Flow-Surface/Result/'+dataset+'-node-features-'+'{:03d}'.format(id)+'-epochs-'+str(itera)+'-samples-'+str(samples)+'-loss-'+str(loss)+'-init-'+str(init)+'.dat',dtype='<f')
		num_of_nodes = len(features)//128
		features = features.reshape(128,num_of_nodes).transpose()
		tsne = manifold.TSNE(n_components=2,n_iter=3000)
		Y = tsne.fit_transform(features)
		clustering = DBSCAN(eps=5, min_samples=1).fit(Y)
		plt.scatter(Y[:,0],Y[:,1],c = getcolor(clustering.labels_))
		plt.show()
		node_matrix = np.fromfile('/Volumes/Research/GNN/'+dataset+'/'+dataset+'/nodes-features-'+'{:03d}'.format(id)+'.dat',dtype='<f')
		node_matrix = node_matrix.reshape(6,num_of_nodes).transpose()
		node_matrix = node_matrix[:,0:3]
		DrawScatter(node_matrix,getcolor(clustering.labels_))
		vertices = getVertices('/Volumes/Research/GNN/'+dataset+'/Quads/'+dataset+'-vertices'+"{:03d}".format(id)+'.txt')
		maps = []
		for j in range(0,len(vertices)):
			dis = np.sum((vertices[j]-node_matrix)**2,axis=1)
			index = np.argmin(dis)
			maps.append(index)
		file = open('/Volumes/Research/GNN/Flow-Surface/TSNE/'+dataset+'-'+"{:03d}".format(id)+'-epochs-'+str(itera)+'-samples-'+str(samples)+'-loss-'+str(loss)+'-init-'+str(init)+'.txt','w')
		for j in range(0,len(Y)):
			file.write(str(Y[j][0]))
			file.write('\t')
			file.write(str(Y[j][1]))
			file.write('\t')
		c = open('/Volumes/Research/GNN/Flow-Surface/TSNE/'+dataset+'-'+"{:03d}".format(id)+'-epochs-'+str(itera)+'-samples-'+str(samples)+'-loss-'+str(loss)+'-init-'+str(init)+'-maps.txt','w')
		for j in range(0,len(maps)):
			c.write(str(maps[j]))
			c.write('\t')
		file.close()
		c.close()

def GetEdgeList(file_path,id,edge_file):
	graph = nx.read_graphml(file_path+'/adjacency-matrix-'+'{:03d}'.format(id)+'.graphml')
	edges = graph.edges
	edge_list = open(edge_file,'w')
	for e in edges:
		edge_list.write(str(int(e[0])+1))
		edge_list.write('\t')
		edge_list.write(str(int(e[1])+1))
		edge_list.write('\n')
	edge_list.close()

def DeepWalkTSNE(embedding,tsne_pos):
	data = open(embedding,'rb')
	d = data.readline()
	s = str(d,'utf-8').strip()
	point = s.split(' ')
	print(point)
	features = np.zeros((int(point[0]),int(point[1])))
	d = data.readline()
	while d:
		s = str(d,'utf-8').strip()
		point = s.split(' ')
		index = int(point[0])
		for j in range(1,len(point)):
			features[index-1][j-1] = float(point[j])
		d = data.readline()
	tsne = manifold.TSNE(n_components=2,n_iter=6000)
	Y = tsne.fit_transform(features)
	file = open(tsne_pos,'w')
	for j in range(0,len(Y)):
		file.write(str(Y[j][0]))
		file.write('\t')
		file.write(str(Y[j][1]))
		file.write('\t')
	file.close()

def GCNTSNE(embedding,tsne_pos,node_matrix_file,node_pos):
	features = np.fromfile(embedding,dtype='<f')
	num_of_nodes = len(features)//128
	features = features.reshape(128,num_of_nodes).transpose()
	tsne = manifold.TSNE(n_components=2,n_iter=6000)
	Y = tsne.fit_transform(features)
	node_matrix = np.fromfile(node_matrix_file,dtype='<f')
	node_matrix = node_matrix.reshape(3,num_of_nodes).transpose()
	vertices = []
	data = open(node_pos,'rb')
	d = data.readline()
	vertices = []
	i = 0
	while d:
		s = str(d,'utf-8').strip()
		point = s.split(' ')
		if point[0] == 'v':
			vertices.append([float(point[1]),float(point[2]),float(point[3])])
		d = data.readline()
	maps = []
	vertices = np.asarray(vertices)
	for j in range(0,len(vertices)):
		dis = np.sum((vertices[j]-node_matrix)**2,axis=1)
		index = np.argmin(dis)
		maps.append(index)
	file = open(tsne_pos,'w')
	for j in range(0,len(Y)):
		file.write(str(Y[j][0]))
		file.write('\t')
		file.write(str(Y[j][1]))
		file.write('\t')
	c = open('/Volumes/Research/GNN/Volumetric/Kidney-080-maps-3000.txt','w')
	for j in range(0,len(maps)):
		c.write(str(maps[j]))
		c.write('\t')
	file.close()
	c.close()

def ReadSurface(path):
  floatSize = 4
  with open(path, "rb") as f:
    numVertices = struct.unpack('i', f.read(floatSize))[0]
    #print(numVertices)
    #vertices = []
    vertices = np.zeros((numVertices,3))
    for i in range(numVertices):
      x = struct.unpack('f', f.read(floatSize))[0]
      y = struct.unpack('f', f.read(floatSize))[0]
      z = struct.unpack('f', f.read(floatSize))[0]
      t = struct.unpack('f', f.read(floatSize))[0]
      #vertices += [x,y,z]
      vertices[i] = np.asarray([x,y,z])
    normals = []
    #normals = np.zeros((numVertices,3))
    for i in range(numVertices):
      x = struct.unpack('f', f.read(floatSize))[0]
      y = struct.unpack('f', f.read(floatSize))[0]
      z = struct.unpack('f', f.read(floatSize))[0]
      normals += [x,y,z]
      #normals[i] = np.asarray([x,y,z])
    num_indices = struct.unpack('i', f.read(floatSize))[0]
    #print(num_indices)
    indices = []
    for i in range(num_indices):
      v1 = struct.unpack('i', f.read(floatSize))[0]
      indices += [v1]
    return vertices,normals,indices

# def GraphEmbed(folder,start,end):
# 	features = np.zeros((end-start,128))
# 	for i in range(start,end):
# 		node_embed = np.fromfile(folder+'40-node-features-'+'{:03d}'.format(i)+'.dat',dtype='<f')
# 		node_embed = node_embed.reshape(128,len(node_embed)//128).transpose()
# 		features[i-start] = np.mean(node_embed,axis=0)
# 	tsne = manifold.TSNE(n_components=2,n_iter=6000)
# 	Y = tsne.fit_transform(features)
# 	file = open('/Volumes/Research/GNN/Scalar/H-40.txt','w')
# 	for j in range(0,len(Y)):
# 		file.write(str(Y[j][0]))
# 		file.write('\t')
# 		file.write(str(Y[j][1]))
# 		file.write('\t')
# 	file.close()
# 	clustering = DBSCAN(eps=12, min_samples=1).fit(Y)
# 	plt.scatter(Y[:,0],Y[:,1],c = getcolor(clustering.labels_))
# 	plt.show()
# features = np.fromfile('/users/vis/Desktop/GNN/5cp-node-features-030.dat',dtype='<f')
# num_of_nodes = len(features)//128
# node_matrix = np.fromfile('/Volumes/LaCie/SurfNet/SurfNet-Result/Data/5cp/nodes-features-030.dat',dtype='<f')
# node_matrix = node_matrix.reshape(6,num_of_nodes).transpose()
# node_matrix = node_matrix[:,0:3]
# print(node_matrix)
# surface_file = '/Volumes/LaCie/SurfNet/SurfNet-Data/5cp/5cp/5cp_surfaces_'+'{:03d}'.format(29)+'.bin'
# vertices,_,_ = ReadSurface(surface_file)
# maps = []
# for j in range(0,len(vertices)):
# 	dis = np.sum((vertices[j]-node_matrix)**2,axis=1)
# 	index = np.argmin(dis)
# 	maps.append(index)
# c = open('/users/vis/Desktop/GNN/5cp-030-maps.txt','w')
# for j in range(0,len(maps)):
# 	c.write(str(maps[j]))
# 	c.write('\t')
# c.close()
'''
for dataset in ['cylinder']:
	for i in [1060,1062]:
		for init in ['pos+vec']:
			for loss in ['shortest']:
				for epochs in [100]:
					for samples in [1000]:
						TSNE(i,dataset,epochs,samples,loss,init)
features = np.zeros((40*4,512))
j = 0
colors = []
k = 0
for var in ['H','He','H+','He+','H2','PD','GT']:
	for i in range(1,41):
		feature = np.fromfile('/Users/vis/Desktop/F/'+var+'-'+'{:04d}'.format(i)+'.dat',dtype='<f')
		features[j] = feature
		j += 1
		colors.append(color_map[k])
	k += 1
tsne = manifold.TSNE(n_components=2,n_iter=6000)
Y = tsne.fit_transform(features)
file = open('/Volumes/Research/V2V-SciVis/Ionization.txt','w')
for j in range(0,len(Y)):
	file.write(str(Y[j][0]))
	file.write('\t')
	file.write(str(Y[j][1]))
	file.write('\t')
file.close()
plt.scatter(Y[:,0],Y[:,1],c=colors)
plt.show()	
GCNTSNE('/Volumes/Research/GNN/Volumetric/Kidney/Kidney-node-features-080-100.dat',
	         '/Volumes/Research/GNN/Volumetric/Kidney-080-SurfNet-3000.txt',
	         '/Volumes/Research/GNN/Volumetric/Kidney/nodes-features-080.dat',
	         '/Volumes/Research/GNN/Volumetric/Kidney/obj/Kidney-080-single.obj')
#DeepWalkTSNE('/Volumes/Research/GNN/Volumetric/Brain/Brain.embeddings','/Volumes/Research/GNN/Volumetric/Brain-035-DeepWalk.txt')
features = np.fromfile('/Volumes/Research/GNN/cylinder/fuse/cylinder-node-features-1062-epochs-100-samples-1000-loss-shortest-init-pos+vec.dat',dtype='<f')
num_of_nodes = len(features)//128
features = features.reshape(128,num_of_nodes).transpose()
tsne = manifold.TSNE(n_components=2,n_iter=6000)
Y = tsne.fit_transform(features)
file = open('/Volumes/Research/GNN/cylinder/cylinder-1062-100-pos+vec.txt','w')
for j in range(0,len(Y)):
	file.write(str(Y[j][0]))
	file.write('\t')
	file.write(str(Y[j][1]))
	file.write('\t')
file.close()
clustering = DBSCAN(eps=12, min_samples=1).fit(Y)
plt.scatter(Y[:,0],Y[:,1],c = getcolor(clustering.labels_))
plt.show()
node_matrix = np.fromfile('/Volumes/Research/GNN/Volumetric/Kidney/nodes-features-080.dat',dtype='<f')
node_matrix = node_matrix.reshape(3,num_of_nodes).transpose()
DrawScatter(node_matrix,getcolor(clustering.labels_))
'''
#GetEdgeList('/Volumes/Research/GNN/Volumetric/Brain',35,'/Volumes/Research/GNN/Volumetric/Brain/Brain.edgelist')