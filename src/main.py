#!/usr/bin/env python3
"""
Pointcloud downsampling with octrees
Octrees calculated by subdividing pointcloud's bounding box
Keep average of points inside each bbox subdivision
"""
import numpy as np
import open3d as o3d
from pointcloud_utils import subdivide, subdivide_recursive, downsample


if __name__ == '__main__':
    # load pointcloud
    pointcloud = o3d.io.read_point_cloud('assets/fragment.ply')
    o3d.visualization.draw_geometries([pointcloud, ])
    exit()

    # crop input pointcloud using eight sub-bboxes
    bbox = pointcloud.get_axis_aligned_bounding_box()
    sub_bboxes = []
    subdivide_recursive(bbox, 5, sub_bboxes)

    # crop pointcloud inside each sub-bbox & calculate their average pointcloud
    pointcloud = downsample(pointcloud, sub_bboxes)

    # draw downsampled pointcloud (with used sub-boxes)
    # o3d.visualization.draw_geometries([pointcloud, ] + sub_bboxes)
    o3d.visualization.draw_geometries([pointcloud, ])
