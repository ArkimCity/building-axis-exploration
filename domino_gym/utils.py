import numpy as np
\
import shapely
from shapely.geometry import LineString, Polygon


def explode_to_segments(curve):
    """Explode a curve into smaller segments

    Args:
        curve (LineString): Curve to explode

    Returns:
        segments(List[Linestring]): Exploded segments that make up the base curve
    """
    curve_coords = list(curve.coords)

    if len(curve_coords) == 0:
        return []

    segments = []
    start_pt = curve_coords[0]

    for i in range(len(curve_coords) - 1):
        end_pt = curve_coords[i + 1]
        segments.append(LineString([start_pt, end_pt]))
        start_pt = end_pt

    return segments




def get_rotated_bb_from_linestring(bb_original: Polygon, line: LineString) -> Polygon:
    vec = (line.coords[1][0] - line.coords[0][0], line.coords[1][1] - line.coords[0][1])
    return get_rotated_bb(bb_original, vec)


def get_rotated_bb(bb_original: Polygon, bb_vec: np.ndarray) -> Polygon:
    """지정된 각도를 가지고 bounding box 를 생성합니다.

    Args:
        bb_original (Polygon): bounding box 를 생성하고자 하는 기준 폴리곤
        bb_vec (np.ndarray): 지정 각도 벡터

    Returns:
        Polygon: 지정된 각도를 가지고 생성한 bounding box
    """
    if bb_original.is_empty:
        return Polygon()

    v = np.array([bb_vec])

    seg_angle = np.degrees(np.arctan2(*v.T[::-1])) % 360.0
    rotated_site_geom = shapely.affinity.rotate(bb_original, -seg_angle[0], origin=bb_original.centroid)
    bb = shapely.geometry.box(*rotated_site_geom.bounds)
    rotated_bb = shapely.affinity.rotate(bb, seg_angle[0], origin=bb_original.centroid)

    return rotated_bb


def get_vec_from_segment(segment: LineString, normalize=False, reverse=False) -> np.ndarray:
    """segment 의 시작점과 종료점을 이용해 np.ndarray 벡터를 생성합니다.

    Args:
        segment (LineString): 벡터를 구하고자 하는 segment
        normalize (bool, optional): True 시 길이를 1로 구합니다. Defaults to False.

    Returns:
        np.ndarray: segment 를 이용해 구한 벡터입니다.
    """
    segment_coords = segment.coords
    start_point = segment_coords[0]
    end_point = segment_coords[-1]

    if reverse:
        start_point, end_point = end_point, start_point

    if normalize:
        xdiff = end_point[0] - start_point[0]
        ydiff = end_point[1] - start_point[1]
        distance = (xdiff**2 + ydiff**2) ** 0.5
        end_point = (start_point[0] + xdiff * (1 / distance), start_point[1] + ydiff * (1 / distance))

    return np.array((end_point[0] - start_point[0], end_point[1] - start_point[1]))
