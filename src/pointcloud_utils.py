import numpy as np
import open3d as o3d

def subdivide_recursive(bbox, n_levels, result):
    """
    Recursive subdivision of pointcloud bbox, `n_levels` times
    # of subdivisions = 8 (for level1) + 8**2 (for level2) + ... + 8**n (for level n)
    """
    if n_levels == 0:
        return

    sub_bboxes = subdivide(bbox)
    result.extend(sub_bboxes)

    for sub_bbox in sub_bboxes:
        subdivide_recursive(sub_bbox, n_levels - 1, result)


def subdivide(bbox):
    # center & bbox radius in all directions
    center = bbox.get_center()
    half_extent = bbox.get_half_extent()
    cx, cy, cz = center[0], center[1], center[2]
    rx, ry, rz = half_extent[0], half_extent[1], half_extent[2]

    # eight sub-bbox from subdivision of pointcloud bbox along xyz axes
    # sub-bbox 1
    sub_bbox1 = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=[cx - rx, cy - ry, cz],
        max_bound=[cx,      cy,      cz + rz]
    )
    sub_bbox1.color = [0, 0, 1]

    # sub-bbox 2
    sub_bbox2 = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=[cx,      cy - ry, cz],
        max_bound=[cx + rx, cy,      cz + rz]
    )
    sub_bbox2.color = [0, 0, 1]

    # sub-bbox 3
    sub_bbox3 = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=[cx - rx, cy,      cz],
        max_bound=[cx,      cy + ry, cz + rz]
    )
    sub_bbox3.color = [0, 0, 1]

    # sub-bbox 4
    sub_bbox4 = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=[cx,      cy,      cz],
        max_bound=[cx + rx, cy + ry, cz + rz]
    )
    sub_bbox4.color = [0, 0, 1]

    # sub-bbox 5
    sub_bbox5 = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=[cx - rx, cy - ry, cz - rz],
        max_bound=[cx,      cy,      cz]
    )
    sub_bbox5.color = [1, 0, 0]

    # sub-bbox 6
    sub_bbox6 = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=[cx,      cy - ry, cz - rz],
        max_bound=[cx + rx, cy,      cz]
    )
    sub_bbox6.color = [1, 0, 0]

    # sub-bbox 7
    sub_bbox7 = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=[cx - rx, cy,      cz - rz],
        max_bound=[cx,      cy + ry, cz]
    )
    sub_bbox7.color = [1, 0, 0]

    # sub-bbox 8
    sub_bbox8 = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=[cx,      cy,      cz - rz],
        max_bound=[cx + rx, cy + ry, cz]
    )
    sub_bbox8.color = [1, 0, 0]

    return [
        sub_bbox1, sub_bbox2, sub_bbox3, sub_bbox4, # row above center
        sub_bbox5, sub_bbox6, sub_bbox7, sub_bbox8, # row below center
    ]

def downsample(pointcloud, sub_bboxes):
    """
    Pointcloud by averaging `pointcloud` points inside each sub-bbox
    """
    pointclouds_avgs = []
    points, colors = [], []

    for sub_bbox in sub_bboxes:
        # crop pointcloud (keep points inside sub-bbox)
        pointcloud_cropped = pointcloud.crop(sub_bbox)
        points_cropped = np.asarray(pointcloud_cropped.points)
        colors_cropped = np.asarray(pointcloud_cropped.colors)

        if len(points_cropped) == 0:
            continue

        # average cropped points
        point_avg = np.average(points_cropped, axis=0)
        color_avg = np.average(colors_cropped, axis=0)
        points.append(point_avg)
        colors.append(color_avg)

    points_vector = o3d.utility.Vector3dVector(np.array(points))
    colors_vector = o3d.utility.Vector3dVector(np.array(colors))
    pointcloud_avg = o3d.geometry.PointCloud(points=points_vector)
    pointcloud_avg.colors = colors_vector

    return pointcloud_avg
