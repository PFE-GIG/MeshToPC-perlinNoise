import sys

def HPR(inputName, outputName):
    import open3d as o3d
    import numpy as np

    pcd = o3d.io.read_point_cloud(inputName)
    diameter = np.linalg.norm(
        np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    o3d.visualization.draw_geometries([pcd])

    camera = [20, 17.5, diameter]
    radius = diameter * 100

    _, pt_map = pcd.hidden_point_removal(camera, radius)

    ##Visualize result
    pcd = pcd.select_by_index(pt_map)
    o3d.visualization.draw_geometries([pcd])

    o3d.io.write_point_cloud(outputName, pcd)

if __name__ == "__main__":
    HPR(str(sys.argv[1]),str(sys.argv[2]))