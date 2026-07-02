import numpy as np
import pyvista as pv
import pandas as pd
import os
import sys

def vtutopython(filename, scalars=[], vectors=[], resample=[], dim=2):

    # filename is the name of the input vtu file (vtk probably works too) in the current directory 
    # scalars is a list of strings containing the names of scalar fields
    # vectors is a list of strings containing the names of vector fields
    # resample is a list of the number of divisions on which to resample the data to get a rectilinear grid, if adaptive meshing was used
    # dim is the dimension of the problem (2 or 3)
    # NOTE: resampling may move the phase field interface slightly

    # returns a dictionary containing numpy arrays of the requested fields

    vecindices = []
    indices = []
    outputs = {}

    mesh = pv.read(os.path.join(os.getcwd(),filename))      # load the mesh

    if len(resample) == 0:
        if dim == 2:
            indexcount = 2

            # put the data into a pandas dataframe to process it (which requires it be loaded into a dictionary first)
            data = {'x': mesh.points[:,0], 'y': mesh.points[:,1]}       
            for field in vectors:
                data[field + 'x'] = mesh.point_data[field][:,0]
                data[field + 'y'] = mesh.point_data[field][:,1]
                vecindices.append((indexcount,field))                   # we will put the fields in a numpy array, so need to know what the indices will be
                indexcount += 2
            for field in scalars:
                data[field] = mesh.point_data[field]
                indices.append((indexcount,field))
                indexcount += 1
            df = pd.DataFrame(data=data)                                # initialize dataframe
            df = df.drop_duplicates()                                   # all values are repeated for each cell node, need to get rid of them
            df = df.sort_values(['x', 'y'], ascending=[True, True])     # sort by position so that we can easily reshape the array
            cleandata = df.to_numpy()                                   # we need to put things in a numpy array so they can be reshaped

            dh = cleandata[1,1]-cleandata[0,1]                          # find spacing between cells
            xsize = cleandata[-1,0]                                     # domain size
            ysize = cleandata[-1,1]
            xdiv = round(xsize/dh)+1                                    # number of points in each direction (number of cells + 1)
            ydiv = round(ysize/dh)+1

            # reshape the arrays and load them into a dictionary to be returned
            outputs['x'] = cleandata[:,0].reshape(xdiv,ydiv)
            outputs['y'] = cleandata[:,1].reshape(xdiv,ydiv)

            for i in indices:
                outputs[i[1]] = cleandata[:,i[0]].reshape(xdiv,ydiv)
            for i in vecindices:
                outputs[i[1]] = np.array([cleandata[:,i[0]].reshape(xdiv,ydiv),cleandata[:,i[0]+1].reshape(xdiv,ydiv)])

            return outputs

        elif dim == 3:
            indexcount = 3

            # put the data into a pandas dataframe to process it (which requires it be loaded into a dictionary first)
            data = {'x': mesh.points[:,0], 'y': mesh.points[:,1], 'z': mesh.points[:,2]}
            for field in vectors:
                data[field + 'x'] = mesh.point_data[field][:,0]
                data[field + 'y'] = mesh.point_data[field][:,1]
                data[field + 'z'] = mesh.point_data[field][:,2]
                vecindices.append((indexcount,field))                               # we will put the fields in a numpy array, so need to know what the indices will be
                indexcount += 3
            for field in scalars:
                data[field] = mesh.point_data[field]
                indices.append((indexcount,field))
                indexcount += 1
            df = pd.DataFrame(data=data)                                            # initialize dataframe
            df = df.drop_duplicates()                                               # all values are repeated for each cell node, need to get rid of them
            df = df.sort_values(['x', 'y', 'z'], ascending=[True, True, True])      # sort by position so that we can easily reshape the array
            cleandata = df.to_numpy()                                               # we need to put things in a numpy array so they can be reshaped

            dh = cleandata[1,2]-cleandata[0,2]                                      # find spacing between cells
            xsize = cleandata[-1,0]                                                 # domain size
            ysize = cleandata[-1,1]
            zsize = cleandata[-1,2]
            xdiv = round(xsize/dh)+1                                                # number of points in each direction (number of cells + 1)
            ydiv = round(ysize/dh)+1
            zdiv = round(zsize/dh)+1

            # reshape the arrays and load them into a dictionary to be returned
            outputs['x'] = cleandata[:,0].reshape(xdiv,ydiv,zdiv)
            outputs['y'] = cleandata[:,1].reshape(xdiv,ydiv,zdiv)
            outputs['z'] = cleandata[:,2].reshape(xdiv,ydiv,zdiv)

            for i in indices:
                outputs[i[1]] = cleandata[:,i[0]].reshape(xdiv,ydiv,zdiv)
            for i in vecindices:
                outputs[i[1]] = np.array([cleandata[:,i[0]].reshape(xdiv,ydiv,zdiv),cleandata[:,i[0]+1].reshape(xdiv,ydiv,zdiv),cleandata[:,i[0]+2].reshape(xdiv,ydiv,zdiv)])

            return outputs

    elif len(resample) == 2:
        indexcount = 2

        mesh0 = pv.create_grid(mesh, dimensions=(resample[0],resample[1],1))
        mesh = mesh0.sample(mesh)

        # put the data into a pandas dataframe to process it (which requires it be loaded into a dictionary first)
        data = {'x': mesh.points[:,0], 'y': mesh.points[:,1]}       
        for field in vectors:
            data[field + 'x'] = mesh.point_data[field][:,0]
            data[field + 'y'] = mesh.point_data[field][:,1]
            vecindices.append((indexcount,field))                   # we will put the fields in a numpy array, so need to know what the indices will be
            indexcount += 2
        for field in scalars:
            data[field] = mesh.point_data[field]
            indices.append((indexcount,field))
            indexcount += 1
        df = pd.DataFrame(data=data)                                # initialize dataframe
        df = df.drop_duplicates()                                   # all values are repeated for each cell node, need to get rid of them
        df = df.sort_values(['x', 'y'], ascending=[True, True])     # sort by position so that we can easily reshape the array
        cleandata = df.to_numpy()                                   # we need to put things in a numpy array so they can be reshaped

        xdiv = resample[0]                                          # number of points in each direction (number of cells + 1)
        ydiv = resample[1]

        # reshape the arrays and load them into a dictionary to be returned
        outputs['x'] = cleandata[:,0].reshape(xdiv,ydiv)
        outputs['y'] = cleandata[:,1].reshape(xdiv,ydiv)

        for i in indices:
            outputs[i[1]] = cleandata[:,i[0]].reshape(xdiv,ydiv)
        for i in vecindices:
            outputs[i[1]] = np.array([cleandata[:,i[0]].reshape(xdiv,ydiv),cleandata[:,i[0]+1].reshape(xdiv,ydiv)])

        return outputs

    elif len(resample) == 3:
        indexcount = 3

        mesh0 = pv.create_grid(mesh, dimensions=(resample[0],resample[1],resample[2]))
        mesh = mesh0.sample(mesh)

        # put the data into a pandas dataframe to process it (which requires it be loaded into a dictionary first)
        data = {'x': mesh.points[:,0], 'y': mesh.points[:,1], 'z': mesh.points[:,2]}
        for field in vectors:
            data[field + 'x'] = mesh.point_data[field][:,0]
            data[field + 'y'] = mesh.point_data[field][:,1]
            data[field + 'z'] = mesh.point_data[field][:,2]
            vecindices.append((indexcount,field))                               # we will put the fields in a numpy array, so need to know what the indices will be
            indexcount += 3
        for field in scalars:
            data[field] = mesh.point_data[field]
            indices.append((indexcount,field))
            indexcount += 1
        df = pd.DataFrame(data=data)                                            # initialize dataframe
        df = df.drop_duplicates()                                               # all values are repeated for each cell node, need to get rid of them
        df = df.sort_values(['x', 'y', 'z'], ascending=[True, True, True])      # sort by position so that we can easily reshape the array
        cleandata = df.to_numpy()                                               # we need to put things in a numpy array so they can be reshaped

        xdiv = resample[0]                                                      # number of points in each direction (number of cells + 1)
        ydiv = resample[1]
        zdiv = resample[2]

        # reshape the arrays and load them into a dictionary to be returned
        outputs['x'] = cleandata[:,0].reshape(xdiv,ydiv,zdiv)
        outputs['y'] = cleandata[:,1].reshape(xdiv,ydiv,zdiv)
        outputs['z'] = cleandata[:,2].reshape(xdiv,ydiv,zdiv)

        for i in indices:
            outputs[i[1]] = cleandata[:,i[0]].reshape(xdiv,ydiv,zdiv)
        for i in vecindices:
            outputs[i[1]] = np.array([cleandata[:,i[0]].reshape(xdiv,ydiv,zdiv),cleandata[:,i[0]+1].reshape(xdiv,ydiv,zdiv),cleandata[:,i[0]+2].reshape(xdiv,ydiv,zdiv)])

        return outputs

if len(sys.argv) > 1:
    fpath=sys.argv[1]
else:
    fpath="solutions/solution_16000000.vtu"

outs = vtutopython(fpath,
                   scalars=['n','x1','x2'],dim=2)

print(outs.keys())
f = open("dealloy_n.in",'wb')
f.write(outs['n'].astype(np.float64).tobytes(order='F'))
f.close()
f = open("dealloy_x1.in",'wb')
f.write(outs['x1'].astype(np.float64).tobytes(order='F'))
f.close()
f = open("dealloy_x2.in",'wb')
f.write(outs['x2'].astype(np.float64).tobytes(order='F'))
f.close()
