import jax.numpy as jnp
import numpy as np
import skfmm

from refineNCBF.utils.types import MaskNd, ArrayNd


# TODO default distance should be .5, since it is from boundary of cell to center of cell


def compute_signed_distance(bool_array: MaskNd) -> ArrayNd:
    signed_distance = jnp.array(-skfmm.distance(~bool_array) + skfmm.distance(bool_array))
    return signed_distance


def expand_mask_by_signed_distance(mask: MaskNd, distance: float = 1) -> MaskNd:
    if np.count_nonzero(mask) == 0 or np.count_nonzero(~mask) == 0:
        print('mask is full, returning mask')
        return mask
    signed_distance = compute_signed_distance(mask)
    expanded_mask = (signed_distance >= -distance)
    return expanded_mask


def shrink_mask_by_signed_distance(mask: MaskNd, distance: float = 1) -> MaskNd:
    if np.count_nonzero(mask) == 0 or np.count_nonzero(~mask) == 0:
        print('mask is full, returning mask')
        return mask
    signed_distance = compute_signed_distance(mask)
    shrunk_mask = (signed_distance >= distance)
    return shrunk_mask


def get_mask_boundary_by_signed_distance(mask: MaskNd, distance: float = 1) -> MaskNd:
    if np.count_nonzero(mask) == 0 or np.count_nonzero(~mask) == 0:
        print('mask is full, returning mask')
        return mask
    signed_distance = compute_signed_distance(mask)
    mask_boundary = (signed_distance <= distance) & (signed_distance >= 0)
    return mask_boundary


def get_mask_boundary_on_both_sides_by_signed_distance(mask: MaskNd, distance: float = 1) -> MaskNd:
    if np.count_nonzero(mask) == 0 or np.count_nonzero(~mask) == 0:
        print('mask is full, returning mask')
        return mask
    signed_distance = compute_signed_distance(mask)
    mask_boundary = (signed_distance <= distance) & (signed_distance >= -distance)
    return mask_boundary
