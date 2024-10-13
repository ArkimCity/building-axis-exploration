import math

import numpy as np
from shapely.geometry import LineString, Point, Polygon


class Plane:
    """origin point, x 방향, y 방향 요소를 가지고 있는 클래스. 도형 생성 기준 혹은 local coordinate system으로 기능."""

    def __init__(
        self,
        origin: np.ndarray = np.array([0.0, 0.0]),
        x_axis: np.ndarray = np.array([1.0, 0.0]),
        y_axis: np.ndarray = np.array([0.0, 1.0]),
    ) -> None:
        """
        각 origin 좌표, X 방향, Y 방향.

        Args:
            origin (np.array, optional): origin 좌표. Defaults to np.array([0.0, 0.0]).
            x_axis (np.array, optional): X 방향. Defaults to np.array([1.0, 0.0]).
            y_axis (np.array, optional): Y 방향. Defaults to np.array([0.0, 1.0]).
        """
        self._origin = origin
        self.__x_axis = x_axis
        self.__y_axis = y_axis

        self.__y_checker_cache = {}

    def __repr__(self) -> str:
        return f"o: {self._origin}, x: {self.__x_axis}, y: {self.__y_axis}"

    def flip_plane_y(self):
        self.__y_axis = -self.__y_axis

    def is_ccw_from_x_to_y(self):
        """외적을 계산해 x축에서 y축으로 시계방향인지 반시계 방향인지 확인합니다.

        Returns:
            bool: 일반적인 경우인 x 축이 수평 오른쪽, y축이 수직 위쪽일 경우 true 입니다.
        """
        determinant = np.cross(self.__x_axis, self.__y_axis)

        # x축과 y축이 Collinear 한 plane 은 잘못된 결과입니다.
        assert math.isclose(determinant, 0, abs_tol=0.01) is False

        if determinant > 0:
            return True

        return False

    @property
    def x_axis(self):
        return self.__x_axis

    @x_axis.setter
    def x_axis(self, value: np.ndarray):
        self.__x_axis = value

    @property
    def y_axis(self):
        return self.__y_axis

    @y_axis.setter
    def y_axis(self, value: np.ndarray):
        self.__y_axis = value

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, value: np.ndarray):
        self._origin = value

    @property
    def origin_point(self):
        return Point(self._origin)

    @property
    def test_geom(self):
        # plot test 확인 용 실제 사용에서는 블러와지면 안됩니다.
        return Polygon(
            [
                self._origin,
                self._origin + self.__x_axis,
                self._origin + self.__x_axis + self.__y_axis,
                self._origin + self.__y_axis,
            ]
        )

    @property
    def unit_polygon(self) -> Polygon:
        # 1값의 정사각형 도형을 반환합니다. normalize 한 x 와 y axis 를 사용합니다.
        x_axis_normalized = self.__x_axis / np.linalg.norm(self.__x_axis)
        y_axis_normalized = self.__y_axis / np.linalg.norm(self.__y_axis)
        return Polygon(
            [
                self._origin,
                self._origin + x_axis_normalized,
                self._origin + x_axis_normalized + y_axis_normalized,
                self._origin + y_axis_normalized,
            ]
        )

    @property
    def y_checker(self) -> LineString:
        cache_key = (tuple(self.origin), tuple(self.origin + self.y_axis * 1000))

        if cache_key not in self.__y_checker_cache:
            self.__y_checker_cache[cache_key] = LineString(cache_key)

        return self.__y_checker_cache[cache_key]

    def get_offset_plane(self, x_offset: float, y_offset: float):
        return Plane(
            origin=self._origin + self.__x_axis * x_offset + self.__y_axis * y_offset,
            x_axis=self.__x_axis,
            y_axis=self.__y_axis,
        )
