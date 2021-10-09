#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numba as numba
import numpy as np
import pims

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def remove_close_corners_flow(bool_field, mask, cs_flow, shape, radius):
    for i, p1 in enumerate(cs_flow):
        x, y = (int(np.round(p1[0][0])), np.round(int(p1[0][1])))
        if not bool_field[x][y]:
            mask[i] = False
            continue
        if i != len(cs_flow) - 1:
            mark_area(x, y, bool_field, shape, radius)


def do_remove_close_corners_flow(bool_field, cs_flow, ids, shape, radius):
    bool_field.fill(True)
    mask = np.ones_like(ids, dtype=bool)
    remove_close_corners_flow(bool_field, mask, cs_flow, shape, radius=radius)
    return cs_flow[mask], ids[mask]


def remove_close_corners_new(bool_field, mask, cs_flow, cs_new, shape, radius):
    for p1 in cs_flow:
        x, y = (int(np.round(p1[0][0])), np.round(int(p1[0][1])))
        mark_area(x, y, bool_field, shape, radius)

    for i, p1 in enumerate(cs_new):
        x, y = (int(np.round(p1[0][0])), np.round(int(p1[0][1])))
        if not bool_field[x][y]:
            mask[i] = False


def do_remove_close_corners_new(bool_field, cs_flow, cs_new, shape, radius):
    bool_field.fill(True)
    mask = np.ones(cs_new.shape[0], dtype=bool)
    remove_close_corners_new(bool_field, mask, cs_flow, cs_new, shape, radius=radius)
    return cs_new[mask]


def mark_area(x, y, bool_field, shape, radius):
    for ax in range(-radius, radius + 1):
        for ay in range(-radius, radius + 1):
            if x + ax in range(shape[1]) and y + ay in range(shape[0]) and np.abs(ax) + np.abs(ay) <= radius:
                bool_field[x + ax][y + ay] = False


mark_area = numba.jit(nopython=True)(mark_area)
remove_close_corners_flow = numba.jit(nopython=True)(remove_close_corners_flow)
remove_close_corners_new = numba.jit(nopython=True)(remove_close_corners_new)


def remove_end_corners(cs_flow, ids, _st):
    return cs_flow[_st.astype(bool)].reshape(-1, 1, 2), ids[_st.reshape(-1).astype(bool)]


def add_new_ids(ids, cur_id, cs_new):
    new_ids = np.arange(cur_id, cur_id + len(cs_new))
    cur_id += len(cs_new)
    return np.concatenate((ids, new_ids))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:

    image_0 = frame_sequence[0]
    corner_size = 11
    new_corner_per_frame = 500
    corner_threshold = 0.01

    cs = cv2.goodFeaturesToTrack(image_0, new_corner_per_frame, corner_threshold, corner_size, blockSize=corner_size)

    cur_id = len(cs)
    ids = np.arange(len(cs))

    corners = FrameCorners(
        ids,
        cs,
        np.ones(len(cs)) * corner_size
    )
    builder.set_corners_at_frame(0, corners)
    bool_field = np.ones((image_0.shape[1], image_0.shape[0]), dtype=bool)

    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        cs_flow, _st, _err = cv2.calcOpticalFlowPyrLK(image_0, image_1, cs, None, winSize=(33, 33), maxLevel=2,
                                                      criteria=(
                                                          cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30,
                                                          1e-2),
                                                      minEigThreshold=0.001)

        cs_flow, ids = remove_end_corners(cs_flow, ids, _st)

        cs_flow, ids = do_remove_close_corners_flow(bool_field, cs_flow, ids, image_1.shape, radius=corner_size)

        cs_new = cv2.goodFeaturesToTrack(image_1, new_corner_per_frame, corner_threshold, corner_size,
                                         blockSize=corner_size)

        cs_new = do_remove_close_corners_new(bool_field, cs_flow, cs_new, image_1.shape, radius=corner_size)

        cs = np.concatenate((cs_flow, cs_new))
        ids = add_new_ids(ids, cur_id, cs_new)
        corners = FrameCorners(
            ids,
            cs,
            np.ones(len(cs)) * corner_size
        )

        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
