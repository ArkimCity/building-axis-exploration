import random
from typing import List, Union

import gym
import numpy as np
from gym import spaces

from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon, LineString

from domino_gym.types import Plane
from domino_gym.utils import get_rotated_bb_from_linestring, explode_to_segments, get_vec_from_segment

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
            y_offset: List[float]
    ):
        """
        Args:
            axis_index: List[int]
            pillar_x_width: List[float]
            pillar_y_width: List[float]
            x_intervals: List[float]
            y_intervals: List[float]
        """
        self.axis_index = axis_index
        self.pillar_x_width = pillar_x_width
        self.pillar_y_width = pillar_y_width
        self.x_intervals = x_intervals
        self.y_intervals = y_intervals
        self.x_offset = x_offset
        self.y_offset = y_offset

    def get_random_action(self):
        return Action(
            axis_index=random.choice(self.axis_index),
            pillar_x_width=random.choice(self.pillar_x_width),
            pillar_y_width=random.choice(self.pillar_y_width),
            x_intervals=random.choices(self.x_intervals, k=10),  # Allow duplicates
            y_intervals=random.choices(self.y_intervals, k=10),  # Allow duplicates
            x_offset=random.uniform(*self.x_offset),
            y_offset=random.uniform(*self.y_offset),
        )

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
        self.__legal_geoms = []

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

    def build(self, action: Action):
        assert self.__is_ready, "Environment is not ready, Please call initialize() first."

        if action.axis_index == 0:
            main_axis = self.__longest_edge_axis
        elif action.axis_index == 1:
            main_axis = self.__obb_axis

        # 정해진 축을 기준으로 시작 지점과 벡터를 지정
        rotated_bb = get_rotated_bb_from_linestring(self.parcel_data, main_axis)
        bb_segments = explode_to_segments(rotated_bb.exterior)

        origin_point = np.array(bb_segments[0].coords[0])
        x_axis = get_vec_from_segment(bb_segments[0], normalize=True)
        y_axis = get_vec_from_segment(bb_segments[-1], normalize=True, reverse=True)

        base_plane = Plane(origin=origin_point, x_axis=x_axis, y_axis=y_axis)

        # action 을 통해 pillars 중심점 지정
