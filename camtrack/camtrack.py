#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import numpy as np
import sortednp as snp
from sklearn import linear_model

from _corners import FrameCorners, StorageImpl
from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    TriangulationParameters,
    triangulate_correspondences,
    rodrigues_and_translation_to_view_mat3x4,
    Correspondences
)


def get2d_3d_correspondence(pts3d, ids, corners):
    ids_1 = ids.flatten()
    ids_2 = corners.ids.flatten()
    _, (indices_1, indices_2) = snp.intersect(ids_1, ids_2, indices=True)
    return pts3d[indices_1], ids_1[indices_1], corners.points[indices_2]


def take_inlier_corners(corners, inl_ids):
    ids_1 = inl_ids.flatten().astype(np.int64)
    ids_1 = np.sort(ids_1)
    ids_2 = corners.ids.flatten()

    _, (indices_1, indices_2) = snp.intersect(ids_1, ids_2, indices=True)
    return FrameCorners(
        ids_2[indices_2],
        corners.points[indices_2],
        corners.sizes[indices_2]
    )


def union_indices(ids1, ids2):
    assert (np.allclose(np.sort(ids1), ids1))
    assert (np.allclose(np.sort(ids2), ids2))
    assert (np.unique(ids1, axis=0, ).shape[0] == ids1.shape[0])
    assert (np.unique(ids2, axis=0, ).shape[0] == ids2.shape[0])

    ids = snp.merge(ids1.flatten(), ids2.flatten(), duplicates=snp.DROP)
    return ids


# пока не используется
def filter_correspondences(corrs):
    cor_vec = corrs.points_2 - corrs.points_1
    ransac = linear_model.RANSACRegressor()
    X = cor_vec[:, 0].reshape(-1, 1)
    y = cor_vec[:, 1]
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    return Correspondences(
        corrs.ids[inlier_mask],
        corrs.points_1[inlier_mask],
        corrs.points_2[inlier_mask]
    )


def do_solve_pnp_ransac(pts3d, pts2d, intrinsic_mat):
    is_success, r_vec, t_vec, inliers = cv2.solvePnPRansac(pts3d, pts2d, intrinsic_mat, np.array([], dtype=float),
                                                           useExtrinsicGuess=False,
                                                           reprojectionError=8.0,
                                                           iterationsCount=100,
                                                           flags=cv2.SOLVEPNP_ITERATIVE)
    if not is_success:
        print('solve pnp failed')
        return None, None

    return rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec.reshape(-1, 1)), inliers


def do_triangulation(corners0, corners1, max_error, min_angle, min_depth, view_mat0, view_mat1, intrinsic_mat):
    corrs = build_correspondences(corners0, corners1)
    tr_params = TriangulationParameters(max_error, min_angle, min_depth)
    pts3d, inl_ids, mean_angle = triangulate_correspondences(corrs, view_mat0,
                                                             view_mat1,
                                                             intrinsic_mat, tr_params)

    return pts3d, inl_ids, mean_angle


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    pts3d, inl_ids, _ = do_triangulation(corner_storage[known_view_1[0]], corner_storage[known_view_2[0]],
                                         2.0, 1.0, 0, pose_to_view_mat3x4(known_view_1[1]),
                                         pose_to_view_mat3x4(known_view_2[1]),
                                         intrinsic_mat)

    point_cloud_builder = PointCloudBuilder(inl_ids,
                                            pts3d)
    frame_count = len(corner_storage)

    view_mats = np.zeros(frame_count, dtype=np.ndarray)
    view_mats[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1])
    view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])

    inl_corners_storage = np.zeros(frame_count, dtype=FrameCorners)
    inl_corners_storage[known_view_1[0]] = take_inlier_corners(corner_storage[known_view_1[0]], inl_ids)
    inl_corners_storage[known_view_2[0]] = take_inlier_corners(corner_storage[known_view_2[0]], inl_ids)

    frame_range = np.arange(0, frame_count)

    inner_mask = np.zeros_like(frame_range, dtype=bool)

    left_known = min(known_view_1[0], known_view_2[0])
    right_known = max(known_view_1[0], known_view_2[0])
    range_between = np.arange(left_known + 1, right_known)
    range_before = np.array([], dtype=int) if left_known == 0 else np.arange(left_known - 1, -1, -1)
    range_after = np.array([], dtype=int) if right_known == frame_count - 1 else np.arange(right_known + 1, frame_count)
    outer_range = np.concatenate((range_between, range_before, range_after))

    for corners_id_1 in outer_range:
        pts3d_1, inl_ids1, pts2d_1 = get2d_3d_correspondence(point_cloud_builder.points, point_cloud_builder.ids,
                                                             corner_storage[corners_id_1])
        view_mat_1, inl_ids_pnp = do_solve_pnp_ransac(pts3d_1, pts2d_1, intrinsic_mat)

        view_mats[corners_id_1] = view_mat_1

        inl_ids_2 = inl_ids1[inl_ids_pnp.flatten()]
        angles = []
        pts_3d_arr = []
        inl_ids_arr = []
        for corners_id_0 in frame_range[inner_mask]:
            pts3d_3, inl_ids_3, mean_cos = do_triangulation(corner_storage[corners_id_0], corner_storage[corners_id_1],
                                                            2.0, 2.0, 0, view_mats[corners_id_0],
                                                            view_mat_1, intrinsic_mat)
            angles.append(np.abs(mean_cos))
            pts_3d_arr.append(pts3d_3)
            inl_ids_arr.append(inl_ids_3)

        tr_points_num = 0
        if len(angles) != 0:
            max_angle_ind = np.argmin(angles)
            inl_ids_2 = union_indices(inl_ids_2, inl_ids_arr[max_angle_ind])
            point_cloud_builder.add_points(inl_ids_arr[max_angle_ind], pts_3d_arr[max_angle_ind])
            tr_points_num = len(inl_ids_arr[max_angle_ind])

        inl_corners = take_inlier_corners(corner_storage[corners_id_1], inl_ids_2)
        inl_corners_storage[corners_id_1] = inl_corners
        inner_mask[corners_id_1] = True
        print(
            f'frame {corners_id_1} : pnp_inliers - {len(inl_ids_pnp)}, triangulated points - {tr_points_num}'
            + f', cloud size - {point_cloud_builder.ids.size}')

    css = StorageImpl(inl_corners_storage)

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        css,
        4.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
