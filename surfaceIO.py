import numpy as np
import struct
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import networkx as nx
import os
from skimage.measure import compare_psnr,compare_nrmse
from skimage.transform import resize
import torch
from math import pi

def ReadHeader(path):
  with open(path, "r") as f:
    dim= f.readline().strip().split() # read first line
    return [int(ele) for ele in dim]

def ReadVectorField(path,dim_x,dim_y,dim_z):
  v=np.fromfile(path,dtype='<f')
  return v.reshape(dim_z,dim_y,dim_x,3).transpose()


def ReadSurface(path):
  floatSize = 4
  with open(path, "rb") as f:
    numVertices = struct.unpack('i', f.read(floatSize))[0]
    #print(numVertices)
    vertices = []
    #vertices = np.zeros((numVertices,4))
    for i in range(numVertices):
      x = struct.unpack('f', f.read(floatSize))[0]
      y = struct.unpack('f', f.read(floatSize))[0]
      z = struct.unpack('f', f.read(floatSize))[0]
      t = struct.unpack('f', f.read(floatSize))[0]
      vertices += [x,y,z]
      #vertices[i] = np.asarray([x,y,z,t])
    print(numVertices)
    print(len(vertices))
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

def ReadSLines(path):
  floatSize = 4
  with open(path, "rb") as f:
    numpoint = struct.unpack('i', f.read(floatSize))[0]
    numstl = struct.unpack('i', f.read(floatSize))[0]
    r = struct.unpack('f', f.read(floatSize))[0]
    tube = np.zeros((128,128,128))
    for i in range(numpoint):
      x = struct.unpack('f', f.read(floatSize))[0]
      y = struct.unpack('f', f.read(floatSize))[0]
      z = struct.unpack('f', f.read(floatSize))[0]
      t = struct.unpack('f', f.read(floatSize))[0]
      tube[int(x)][int(y)][int(z)] = 1
      tube[int(x)+1][int(y)+1][int(z)+1] = 1
      tube[int(x)][int(y)+1][int(z)] = 1
      tube[int(x)+1][int(y)][int(z)] = 1
      tube[int(x)][int(y)][int(z)+1] = 1
      tube[int(x)+1][int(y)+1][int(z)] = 1
      tube[int(x)][int(y)+1][int(z)+1] = 1
      tube[int(x)+1][int(y)][int(z)+1] = 1
    return np.sum(tube)/(128*128*128)

def ReadPLines(path):
  floatSize = 4
  with open(path, "rb") as f:
    numpoint = struct.unpack('i', f.read(floatSize))[0]
    numstl = struct.unpack('i', f.read(floatSize))[0]
    tube = np.zeros((10,128,128,128))
    times = []
    for i in range(numpoint):
      x = struct.unpack('f', f.read(floatSize))[0]
      y = struct.unpack('f', f.read(floatSize))[0]
      z = struct.unpack('f', f.read(floatSize))[0]
      t = struct.unpack('f', f.read(floatSize))[0]
      tube[int(t)][int(x)][int(y)][int(z)] = 1
      tube[int(t)][int(x)+1][int(y)+1][int(z)+1] = 1
      tube[int(t)][int(x)+1][int(y)+1][int(z)+1] = 1
      tube[int(t)][int(x)][int(y)+1][int(z)] = 1
      tube[int(t)][int(x)+1][int(y)][int(z)] = 1
      tube[int(t)][int(x)][int(y)][int(z)+1] = 1
      tube[int(t)][int(x)+1][int(y)+1][int(z)] = 1
      tube[int(t)][int(x)][int(y)+1][int(z)+1] = 1
      tube[int(t)][int(x)+1][int(y)][int(z)+1] = 1

      tube[int(t)+1][int(x)][int(y)][int(z)] = 1
      tube[int(t)+1][int(x)+1][int(y)+1][int(z)+1] = 1
      tube[int(t)+1][int(x)+1][int(y)+1][int(z)+1] = 1
      tube[int(t)+1][int(x)][int(y)+1][int(z)] = 1
      tube[int(t)+1][int(x)+1][int(y)][int(z)] = 1
      tube[int(t)+1][int(x)][int(y)][int(z)+1] = 1
      tube[int(t)+1][int(x)+1][int(y)+1][int(z)] = 1
      tube[int(t)+1][int(x)][int(y)+1][int(z)+1] = 1
      tube[int(t)+1][int(x)+1][int(y)][int(z)+1] = 1
      '''
    print(np.max(times))
    print(np.min(times))

    return [np.sum(tube[j])/(128*128*128) for j in range(0,10)]

def WriteSurface(path,vertices,normals,indices):
  print(indices)
  surface = open(path, 'wb')
  surface.write(struct.pack('i', len(normals)//3))
  for i in range(0,len(vertices),4):
    surface.write(struct.pack('f',vertices[i]))
    surface.write(struct.pack('f',vertices[i+1]))
    surface.write(struct.pack('f',vertices[i+2]))
    surface.write(struct.pack('f',vertices[i+3]))
  for i in range(0,len(normals),3):
    surface.write(struct.pack('f',normals[i]))
    surface.write(struct.pack('f',normals[i+1]))
    surface.write(struct.pack('f',normals[i+2]))
  surface.write(struct.pack('i', len(indices)))
  for i in range(0,len(indices)):
    surface.write(struct.pack('i',indices[i]))
    #surface.write(struct.pack('i',indices[i+1]))
    #surface.write(struct.pack('i',indices[i+2]))
    #surface.write(struct.pack('i',indices[i]))
    #surface.write(struct.pack('i',indices[i+2]))
    #surface.write(struct.pack('i',indices[i+3]))

def ConvertSurfaceFromObjToDat(obj,dat,write_path):
  data = open(obj,'rb')
  d = data.readline()
  i = 0
  indices = []
  while d:
    s = str(d,'utf-8').strip()
    point = s.split(' ')
    if point[0] == 'v':
      continue
    elif point[0] == 'f':
      if '/' in point[1]:
        p1 = point[1][:point[1].find("/")]
        p2 = point[2][:point[2].find("/")]
        p3 = point[3][:point[3].find("/")]
      else:
        p1 = point[1]
        p2 = point[2]
        p3 = point[3]
      indices += [int(p1)-1,int(p2)-1,int(p3)-1]
    d = data.readline()
  vertices,normals,_ = ReadSurface(dat)
  WriteSurface(write_path,vertices,normals,indices)



def GenerateMesh(file,vertices,faces,num):
  obj = open(file,'w')
  for i in range(0,len(vertices),3):
    obj.write('v')
    obj.write(' ')
    obj.write(str(vertices[i]))
    obj.write(' ')
    obj.write(str(vertices[i+1]))
    obj.write(' ')
    obj.write(str(vertices[i+2]))
    obj.write('\n')
  if num == 3:
    for i in range(0,len(faces),3):
      obj.write('f')
      obj.write(' ')
      obj.write(str(faces[i]+1))
      obj.write(' ')
      obj.write(str(faces[i+1]+1))
      obj.write(' ')
      obj.write(str(faces[i+2]+1))
      obj.write('\n')
  elif num == 4:
    for i in range(0,len(faces),4):
      obj.write('f')
      obj.write(' ')
      obj.write(str(faces[i]+1))
      obj.write(' ')
      obj.write(str(faces[i+1]+1))
      obj.write(' ')
      obj.write(str(faces[i+2]+1))
      obj.write('\n')

      obj.write('f')
      obj.write(' ')
      obj.write(str(faces[i]+1))
      obj.write(' ')
      obj.write(str(faces[i+3]+1))
      obj.write(' ')
      obj.write(str(faces[i+2]+1))
      obj.write('\n')
  obj.close()

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

def RepIS():
  IS = {'H':[151,236,40],'He':[46,116,222],'PD':[39,105,192,5],'MF':[69,161,240],'HR':[5,51,92,194],'YOH':[13,84,170]}
  Surf = {'H':[147,254,37],'He':[220,151,110],'PD':[43,103,91,6],'MF':[81,158,234],'HR':[125,88,48,3],'YOH':[163,80,31]}
  Id = {'H':40, 'MF': 30, 'HR':50,'He':60,'PD':60,'YOH':30}
  for var in ['YOH']:
    pos = open('/Volumes/Research/GNN/Scientific/'+var+'-'+str(Id[var])+'-pos-IS.txt','w')
    value = open('/Volumes/Research/GNN/Scientific/'+var+'-'+str(Id[var])+'-isov-IS.txt','w')
    for j in IS[var]:
      if var in ['H','He','PD']:
        surface_file = '/Volumes/Research/GNN/Scientific/Ionization/'+var+'/isosurfaces/'+str(Id[var])+'/'+var+'-'+'{:01d}'.format(j)+'.dat'
      else:
        surface_file = '/Volumes/Research/GNN/Scientific/Combustion/'+var+'/isosurfaces/'+str(Id[var])+'/'+var+'-'+'{:01d}'.format(j)+'.dat'
      vertices,_,_, = ReadSurface(surface_file)
      for k in range(0,len(vertices)):
        pos.write(str(vertices[j]))
        pos.write('\t')
        value.write(str(j))
        value.write('\t')
    pos.close()
    value.close()


def RepSS():
  Flow = {'ttornado':[808,634,87,45],'tornado':[2139,2398,2379,2592],'5cp':[110,476,294,551,247,802,116],'cylinder':[253,805,235],'plume':[242,95,851,783,891,360,863]}
  Surf = {'ttornado':[75,495,152,142],'tornado':[439,998,970,450],'5cp':[420,84,715,555,336,279,886],'cylinder':[634,288,443],'benard':[7,55,161,734]}
  dim = {'ttornado':[64,64,64],'tornado':[64,64,64],'5cp':[51,51,51],'cylinder':[192,64,48],'benard':[128,32,64],'plume':[126,126,512]}
  for var in ['plume']:
    vec = np.fromfile('/Volumes/Research/GNN/vectors/'+var+'.vec',dtype='<f')
    vec = vec.reshape(dim[var][2],dim[var][1],dim[var][0],3).transpose()
    pos = open('/Volumes/Research/GNN/Flow/'+var+'-pos-Flow.txt','w')
    value = open('/Volumes/Research/GNN/Flow/'+var+'-vec-Flow.txt','w')
    for j in Flow[var]:
      surface_file = '/Volumes/Research/GNN/'+var+'/surfaces/'+var+'_surfaces_'+'{:03d}'.format(j)+'.bin'
      vertices,_,_, = ReadSurface(surface_file)
      for k in range(0,len(vertices),3):
        pos.write(str(vertices[k]))
        pos.write('\t')
        pos.write(str(vertices[k+1]))
        pos.write('\t')
        pos.write(str(vertices[k+2]))
        pos.write('\t')
        judge,velocity = GetVelocity(vec,[vertices[k],vertices[k+1],vertices[k+2]],dim[var][0],dim[var][1],dim[var][2])
        if judge:
          value.write(str(velocity[0]))
          value.write('\t')
          value.write(str(velocity[1]))
          value.write('\t')
          value.write(str(velocity[2]))
          value.write('\t')
        else:
          value.write(str(0))
          value.write('\t')
          value.write(str(0))
          value.write('\t')
          value.write(str(0))
          value.write('\t')
    pos.close()
    value.close()


def EvalutionISO():
  Id = {'H':40, 'MF': 30, 'HR':50,'He':60,'PD':60,'YOH':30}
  for var in ['H','MF','PD','He','HR','YOH']:
    for app in ['IS','Surf']:
      if var in ['H','He','PD']:
        gt = np.fromfile('/Volumes/Research/GNN/Scientific/Ionization/'+var+'/Data/'+var+'{:04d}'.format(Id[var])+'.dat',dtype='<f')
        res = np.fromfile('/Volumes/Research/GNN/vectors/'+var+'-'+'{:01d}'.format(Id[var])+'-'+app+'.dat',dtype='<f')
        gt = gt.reshape(124,124,300).transpose()
        res = res.reshape(124,124,300,3).transpose()
      elif var in ['MF','HR','YOH']:
        gt = np.fromfile('/Volumes/Research/GNN/Scientific/Combustion/'+var+'/Data/'+var+'{:04d}'.format(Id[var])+'.dat',dtype='<f')
        res = np.fromfile('/Volumes/Research/GNN/vectors/'+var+'-'+'{:01d}'.format(Id[var])+'-'+app+'.dat',dtype='<f')
        gt = gt.reshape(60,360,240).transpose()
        res = res.reshape(60,360,240,3).transpose()
      '''
      # print(np.max(gt))
      # print(np.min(gt))
      # print(np.max(res))
      # print(np.min(res))
      '''
      print(var+'-'+app)
      r = compare_nrmse(gt,res[0])
      print('RMSE = '+str(r))
      r = compare_psnr(gt,res[0],data_range=255)
      print('PSNR = '+str(r))

def getAAD(t,v):
  
  t = torch.FloatTensor(t)
  v = torch.FloatTensor(v)
  cos = torch.sum(t*v,dim=0) / (torch.norm(t, dim=0) * torch.norm(v, dim=0) + 1e-10)
  cos[cos>1] = 1
  cos[cos<-1] = -1
  aad = torch.mean(torch.acos(cos)).item() / pi
  return aad

def EvalutionSS():
  dim = {'tornado':[64,64,64],'tornado':[64,64,64],'5cp':[51,51,51],'cylinder':[192,64,48],'benard':[128,32,64],'plume':[126,126,512]}
  for var in ['plume']:
    for app in ['Flow']:
      gt = np.fromfile('/Volumes/Research/GNN/vectors/'+var+'.vec',dtype='<f')
      res = np.fromfile('/Volumes/Research/GNN/vectors/'+var+'-'+app+'.vec',dtype='<f')
      gt = gt.reshape(dim[var][2],dim[var][1],dim[var][0],3).transpose()
      res = res.reshape(dim[var][2],dim[var][1],dim[var][0],3).transpose()
      print(var+'-'+app)
      m = np.max(gt)-np.min(gt)
      r = compare_psnr(gt,res,data_range=m)
      print('PSNR = '+str(r))
      r = getAAD(gt,res)
      print('AAD = '+str(r))

#EvalutionSS()
#RepSS()

# 
# for j in range(1,2):
#   #print(j)
#   line = '/users/vis/Desktop/I/tornado-1-10-30000.ptl'
#   p = ReadPLines(line)
'''

