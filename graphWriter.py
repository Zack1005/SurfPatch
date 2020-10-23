
import argparse
import networkx as nx
import SurfaceProcesser as sp
import surfaceIO
import numpy as np
#import GNN
parser = argparse.ArgumentParser(description='patch generator')
parser.add_argument('--case_num', type=int, default=4000,
                    help='number of train size')
parser.add_argument('--dataset', type=str,
                    help='')
args = parser.parse_args()

def main():

    hdr_path = './vector_field/' + args.dataset + '.hdr'
    dim = surfaceIO.ReadHeader(hdr_path)

    # vec_field: a 3D array
    vec_field_path = './vector_field/' + args.dataset + '.vec'
    vec_field = surfaceIO.ReadVectorField(vec_field_path, dim[0], dim[1], dim[2])

    idx = 0

    while idx < args.case_num:
        caseId = "{0:0=3d}".format(idx)  # to form idx to three digits
        surface_file_path = "./data/" + args.dataset + "/surface_files/"+args.dataset+"_surfaces_" + caseId + ".bin"
        vertices, normals, indices = surfaceIO.ReadSurface(surface_file_path)
        if(idx%100==0):
            print(surface_file_path)
        adjacent_matrix_file_path = "./data/" + args.dataset + "/extracted_files/adjacent_matrix_" + caseId + ".graphml"
        node_features_file_path = "./data/" + args.dataset + "/extracted_files/node_features_" + caseId + ".bin"

        sp.WriteGraph(vertices, indices, vec_field, dim[0], dim[1], dim[2], adjacent_matrix_file_path,
                      node_features_file_path)
        idx += 1

if __name__ == "__main__":
    main()
