
import numpy as np

def triangleArea(v1,v2,v3):
    print(v1)
    return 0.5 * np.linalg.norm(np.cross(v2 - v1 , v3 - v1),axis=1)

def meshtoPc(inputName, outputName1, outputName2):
    import open3d as o3d
    import pandas as pd
    mesh = o3d.io.read_triangle_mesh(inputName)
    n = 100000

    V1id = np.asarray(mesh.triangles)[:,0]
    V2id = np.asarray(mesh.triangles)[:,1]
    V3id = np.asarray(mesh.triangles)[:,2]
    np.asarray(mesh.vertices)[V1id[0],0]
    #print(V1id[0])
    #print(np.asarray(mesh.vertices)[V1id[0]])
    #print(np.asarray(mesh.vertices)[V1id[0],0])

    rows,cols = np.asarray(mesh.triangles).shape

    v1 = []
    v2 = []
    v3 = []
    for i in range(0,rows):
        v1 = np.append(v1, np.asarray(mesh.vertices)[V1id[i]],axis=0)
        v2 = np.append(v2, np.asarray(mesh.vertices)[V2id[i]],axis=0)
        v3 = np.append(v3, np.asarray(mesh.vertices)[V3id[i]],axis=0)
    v1 = np.resize(v1,(rows,3))
    v2 = np.resize(v2,(rows,3))
    v3 = np.resize(v3,(rows,3))
    print(np.asarray(mesh.triangles).shape)
    print(v1,v2,v3)

    areas = triangleArea(v1,v2,v3)
    probabilities = areas / areas.sum()
    weightedRndId = np.random.choice(range(len(areas)), size=n, p=probabilities)

    v1w = v1[weightedRndId]
    v2w = v2[weightedRndId]
    v3w = v3[weightedRndId]

    u = np.random.rand(n,1)
    v = np.random.rand(n,1)

    overOne = u + v > 1
    u[overOne] = 1 - u[overOne]
    v[overOne] = 1 - v[overOne]

    w = 1 - (u + v)

    result = pd.DataFrame()

    result_xyz = (v1w * u) + (v2w * v) + (v3w * w)
    result_xyz = result_xyz.astype(np.float32)

    result["x"] = result_xyz[:,0]
    result["y"] = result_xyz[:,1]
    result["z"] = result_xyz[:,2]

    print(result_xyz)
    result.head()
    from pyntcloud.io import write_ply
    write_ply(outputName1, points = result)

    #Noised PC
    from perlin_noise import PerlinNoise
    noise = PerlinNoise(octaves=10, seed=1)
    noisedPC = []
    noisedResult = pd.DataFrame()

    A = v2w - v1w
    B = v3w - v1w

    for i in range(0,v1w.shape[0]-1):
        Nx = A[i,1] * B[i,2] - A[i,2] * B[i,1]
        Ny = A[i,2] * B[i,0] - A[i,0] * B[i,2]
        Nz = A[i,0] * B[i,1] - A[i,1] * B[i,0]
        noiseVal = noise(result_xyz[i,:])
        #print("noisecal", noiseVal)
        #print(result_xyz[i,:])
        result_xyz[i,:] = result_xyz[i,:] + np.array([Nx, Ny, Nz]) * noiseVal*0.05
        #print(result_xyz[i,:])

    noisedResult["x"] = result_xyz[:,0]
    noisedResult["y"] = result_xyz[:,1]
    noisedResult["z"] = result_xyz[:,2]
    noisedResult.head()

    write_ply(outputName2, points = noisedResult)

if __name__ == "__main__":
    meshtoPc(str(sys.argv[1]),str(sys.argv[2]),str(sys.argv[3]))