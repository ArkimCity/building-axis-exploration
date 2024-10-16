import random
from typing import List, Union, Tuple

import gym
import numpy as np
from gym import spaces

from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon, LineString

from domino_gym.types import Plane
from domino_gym.utils import get_rotated_bb_from_linestring, explode_to_segments, get_vec_from_segment, make_pillar_polygon

import debugvisualizer as dv  # pylint: disable=unused-import


class LawSettings:
    def __init__(
            self,
            openspace_width: float,
            sunlight_cut_start_height: float,
            sunlight_cut_ratio: float,
            floor_height: float,
            max_floor_count: int
    ):
        """
        Args:
            openspace_width: float
            sunlight_cut_start_height: float
            sunlight_cut_ratio: float
            floor_height: float
            max_floor_count: int
        """
        self.openspace_width = openspace_width
        self.sunlight_cut_start_height = sunlight_cut_start_height
        self.sunlight_cut_ratio = sunlight_cut_ratio
        self.floor_height = floor_height
        self.max_floor_count = max_floor_count


class Action:
    def __init__(
            self,
            axis_index: int,
            pillar_x_width: float,
            pillar_y_width: float,
            x_intervals: float,
            y_intervals: float,
            x_offset: float,
            y_offset: float
    ):
        self.axis_index = axis_index
        self.pillar_x_width = pillar_x_width
        self.pillar_y_width = pillar_y_width
        self.x_intervals = x_intervals
        self.y_intervals = y_intervals
        self.x_offset = x_offset
        self.y_offset = y_offset

class SearchSapce:
    def __init__(
            self,
            axis_index: List[int],
            pillar_x_width: List[float],
            pillar_y_width: List[float],
            x_intervals: List[float],
            y_intervals: List[float],
            x_offset: List[float],
            y_offset: List[float],
            x_rows_max: int,
            y_rows_max: int
    ):
        """
        Args:
            axis_index: List[int]
            pillar_x_width: List[float]
            pillar_y_width: List[float]
            x_intervals: List[float]
            y_intervals: List[float]
            x_rows_max: int
            y_rows_max: int
        """
        self.axis_index = axis_index
        self.pillar_x_width = pillar_x_width
        self.pillar_y_width = pillar_y_width
        self.x_intervals = x_intervals
        self.y_intervals = y_intervals
        self.x_rows_max = x_rows_max
        self.y_rows_max = y_rows_max
        self.x_offset = x_offset
        self.y_offset = y_offset

    def get_random_action(self):
        return Action(
            axis_index=random.choice(self.axis_index),
            pillar_x_width=random.choice(self.pillar_x_width),
            pillar_y_width=random.choice(self.pillar_y_width),
            x_intervals=random.choices(self.x_intervals, k=self.x_rows_max),
            y_intervals=random.choices(self.y_intervals, k=self.y_rows_max),
            x_offset=random.uniform(*self.x_offset),
            y_offset=random.uniform(*self.y_offset),
        )


class BuildResult:
    def __init__(
            self,
            base_plane: Plane,
            offset_plane: Plane,
            pillar_centers: np.ndarray,
            pillar_polygons: List[List[Polygon]],
            valid_pairs_all_floors: List[List[Tuple[int, int]]],
            valid_pillar_centers_all_floors: List[np.ndarray],
            valid_pillar_polygons_all_floors: List[List[Polygon]],
            valid_slab_patches_all_floors: List[List[Polygon]]
    ):
        self.__base_plane = base_plane
        self.__offset_plane = offset_plane
        self.__pillar_centers = pillar_centers
        self.__pillar_polygons = pillar_polygons
        self.__valid_pairs_all_floors = valid_pairs_all_floors
        self.__valid_pillar_centers_all_floors = valid_pillar_centers_all_floors
        self.__valid_pillar_polygons_all_floors = valid_pillar_polygons_all_floors
        self.__valid_slab_patches_all_floors = valid_slab_patches_all_floors
        self.__total_area = sum(x.area for y in self.__valid_slab_patches_all_floors for x in y)

    @property
    def base_plane(self):
        return self.__base_plane

    @property
    def offset_plane(self):
        return self.__offset_plane

    @property
    def pillar_centers(self):
        return self.__pillar_centers

    @property
    def pillar_polygons(self):
        return self.__pillar_polygons

    @property
    def valid_pairs_all_floors(self):
        return self.__valid_pairs_all_floors

    @property
    def valid_pillar_centers_all_floors(self):
        return self.__valid_pillar_centers_all_floors

    @property
    def valid_pillar_polygons_all_floors(self):
        return self.__valid_pillar_polygons_all_floors

    @property
    def valid_slab_patches_all_floors(self):
        return self.__valid_slab_patches_all_floors

    @property
    def total_area(self):
        return self.__total_area


class ParcelEnv(gym.Env):
    def __init__(self, parcel_data, search_settings: SearchSapce, law_settings: LawSettings):
        """
        Args:
            parcel_data: shapely.geometry.Polygon
            search_settings: SearchSapce
            law_settings: LawSettings
        """
        super(ParcelEnv, self).__init__()
        self.__is_ready = False

        self.__search_settings = search_settings
        self.__law_settings = law_settings

        # 필지 데이터를 저장
        self.parcel_data = parcel_data
        self.__legal_geoms: List[Union[Polygon, MultiPolygon]] = []
        self.__longest_edge_axis: LineString = None
        self.__obb_axis: LineString = None

        # FIXME: AXIS 방향에 따른 최대 개수가 있긴 함 - 10 은 임시값
        parameter_shape = [10, 10]
        self.action_space = spaces.Box(low=0, high=1, shape=parameter_shape, dtype=np.float32)

    @property
    def search_settings(self):
        return self.__search_settings

    @property
    def law_settings(self):
        return self.__law_settings

    @property
    def legal_geoms(self) -> List[Union[Polygon, MultiPolygon]]:
        return self.__legal_geoms

    @property
    def longest_edge_axis(self) -> LineString:
        return self.__longest_edge_axis

    @property
    def obb_axis(self) -> LineString:
        return self.__obb_axis

    def initialize(self):
        legal_geoms = []

        legal_geom_base = self.parcel_data.buffer(-self.law_settings.openspace_width)
        for floor_idx in range(self.law_settings.max_floor_count):

            current_height = (floor_idx + 1) * self.law_settings.floor_height
            if current_height < self.law_settings.sunlight_cut_start_height:
                dy = 0
            else:
                dy = -current_height * self.law_settings.sunlight_cut_ratio

            parcel_shifted_y = translate(self.parcel_data, yoff=dy)
            each_legal_geom = legal_geom_base.intersection(parcel_shifted_y)

            legal_geoms.append(each_legal_geom)

        self.__legal_geoms = legal_geoms

        parcel_segments = explode_to_segments(self.parcel_data.exterior)
        self.__longest_edge_axis = max(parcel_segments, key=lambda x: x.length)
        self.__obb_axis = LineString(
            [
                self.parcel_data.minimum_rotated_rectangle.exterior.coords[0],
                self.parcel_data.minimum_rotated_rectangle.exterior.coords[1],
            ]
        )

        self.__is_ready = True

    def make_base(self, action: Action):
        if action.axis_index == 0:
            main_axis = self.longest_edge_axis
        elif action.axis_index == 1:
            main_axis = self.obb_axis

        # 정해진 축을 기준으로 시작 지점과 벡터를 지정
        rotated_bb = get_rotated_bb_from_linestring(self.parcel_data, main_axis)
        bb_segments = explode_to_segments(rotated_bb.exterior)

        origin_point = np.array(bb_segments[0].coords[0])
        x_axis = get_vec_from_segment(bb_segments[0], normalize=True)
        y_axis = get_vec_from_segment(bb_segments[-1], normalize=True, reverse=True)

        base_plane = Plane(origin=origin_point, x_axis=x_axis, y_axis=y_axis)

        # action 을 통해 pillars 중심점 지정
        offset_plane = base_plane.get_offset_plane(action.x_offset, action.y_offset)

        # 중심점을 기준으로 각 pillars 의 위치 지정
        x_locations = [sum(action.x_intervals[:i]) for i, _ in enumerate(action.x_intervals)]
        y_locations = [sum(action.y_intervals[:i]) for i, _ in enumerate(action.y_intervals)]

        # make 3d array with x_locations, y_locations - shape: (len(x_locations), len(y_locations), 2)
        pillar_centers: np.ndarray = np.array([
            [
                offset_plane.origin + offset_plane.x_axis * x_location + offset_plane.y_axis * y_location
                for y_location in y_locations
            ]
            for x_location in x_locations
        ])

        pillar_polygons: List[List[Polygon]] = [
            [
                make_pillar_polygon(
                    pillar_center,
                    offset_plane.x_axis,
                    offset_plane.y_axis,
                    action.pillar_x_width,
                    action.pillar_y_width
                )
                for pillar_center in _
            ]
            for _ in pillar_centers
        ]

        return base_plane, offset_plane, pillar_centers, pillar_polygons

    def find_valid_pillars(self, pillar_centers: np.ndarray, pillar_polygons: List[List[Polygon]]):

        valid_pairs_all_floors: List[List[Tuple[int, int]]] = []
        valid_pillar_centers_all_floors: List[np.ndarray] = []
        valid_pillar_polygons_all_floors: List[List[Polygon]] = []
        valid_slab_patches_all_floors: List[List[Polygon]] = []

        for floor_idx, each_legal_geom in enumerate(self.legal_geoms):
            pillar_polygon_within_bool_map = np.array([
                [
                    pillar_polygon.within(each_legal_geom)
                    for pillar_polygon in each_pillar_polygons
                ]
                for each_pillar_polygons in pillar_polygons
            ])

            valid_pairs = []
            valid_slab_patches: List[Polygon] = []
            for i in range(len(pillar_polygon_within_bool_map) - 1):
                for j in range(len(pillar_polygon_within_bool_map[0]) - 1):
                    pairs = [(i + dx, j + dy) for dx in (0, 1) for dy in (0, 1)]

                    # 모두 법규 안에 포함되고, 아래층에서 올라올 수 있는 경우 허용
                    if all([pillar_polygon_within_bool_map[x][y] for x, y in pairs]) and (floor_idx == 0 or all([
                        pair in valid_pairs_all_floors[floor_idx - 1] for pair in pairs
                    ])):
                        valid_slab_patches.append(Polygon([
                            pillar_centers[i][j],
                            pillar_centers[i + 1][j],
                            pillar_centers[i + 1][j + 1],
                            pillar_centers[i][j + 1]
                        ]))
                        valid_pairs.extend(pairs)
            valid_pairs = list(set(valid_pairs))

            valid_pillar_centers = [pillar_centers[i][j] for i, j in valid_pairs]
            valid_pillar_polygons = [pillar_polygons[i][j] for i, j in valid_pairs]

            valid_pairs_all_floors.append(valid_pairs)
            valid_pillar_centers_all_floors.append(valid_pillar_centers)
            valid_pillar_polygons_all_floors.append(valid_pillar_polygons)
            valid_slab_patches_all_floors.append(valid_slab_patches)

        return (
            valid_pairs_all_floors,
            valid_pillar_centers_all_floors,
            valid_pillar_polygons_all_floors,
            valid_slab_patches_all_floors
        )

    def build(self, action: Action) -> BuildResult:
        assert self.__is_ready, "Environment is not ready, Please call initialize() first."

        base_plane, offset_plane, pillar_centers, pillar_polygons = self.make_base(action)
        (
            valid_pairs_all_floors,
            valid_pillar_centers_all_floors,
            valid_pillar_polygons_all_floors,
            valid_slab_patches_all_floors
        ) = self.find_valid_pillars(pillar_centers, pillar_polygons)

        return BuildResult(
            base_plane=base_plane,
            offset_plane=offset_plane,
            pillar_centers=pillar_centers,
            pillar_polygons=pillar_polygons,
            valid_pairs_all_floors=valid_pairs_all_floors,
            valid_pillar_centers_all_floors=valid_pillar_centers_all_floors,
            valid_pillar_polygons_all_floors=valid_pillar_polygons_all_floors,
            valid_slab_patches_all_floors=valid_slab_patches_all_floors
        )
