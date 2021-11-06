#!/usr/bin/env python3
"""
Pointcloud downsampling with octrees
Octrees calculated by manually subdividing pointcloud's bounding box
Keep average of points inside each bbox subdivision
"""
import numpy as np
import open3d as o3d
from pointcloud_utils import subdivide, downsample


if __name__ == '__main__':
    # load pointcloud
    # pointcloud = o3d.io.read_point_cloud('assets/bunny.ply')
    pointcloud = o3d.io.read_point_cloud('assets/fragment.ply')
    points = np.asarray(pointcloud.points)
    print('type points: {}'.format(type(pointcloud.points)))
    print('points: {}'.format(np.asarray(pointcloud.points)))
    print()

    # crop input pointcloud using eight sub-bboxes
    sub_bboxes = subdivide(pointcloud)

    """
    # show pointcloud
    sub_bboxes = [
        sub_bboxes[0], sub_bboxes[1], sub_bboxes[2], sub_bboxes[3], # row above center
        sub_bboxes[4], sub_bboxes[5], sub_bboxes[6], sub_bboxes[7], # row below center
    ]
    """

    # crop pointcloud inside each sub-bbox & calculate their average pointcloud
    # geometries = downsample(pointcloud, sub_bboxes)

    # geometries = [pointcloud,]
    o3d.visualization.draw_geometries([pointcloud, ] + sub_bboxes)
    # o3d.visualization.draw_geometries(geometries + sub_bboxes)
