import random
import numpy as np
from shapely.io import from_wkt

import debugvisualizer as dv  # pylint: disable=unused-import
from domino_gym.env import ParcelEnv, LawSettings, SearchSapce

PARCEL_EXAMPLES = [
    "POLYGON((192670.92948224672 442683.615937913593,192673.509890760877 442672.811465914885,192692.349158951663 442677.159184522519,192689.87916297061 442687.849398120306,192670.92948224672 442683.615937913593)))",
]

EPOCHS = 10

if __name__ == "__main__":
    parcel_data = from_wkt(PARCEL_EXAMPLES[0])

    axis_index = (0, 1)
    pillar_x_width = (0.2,)
    pillar_y_width = (0.4,)
    x_intervals = (2.0, 6.0)
    y_intervals = (6.0,)
    x_offset = (0.0, 10.0)
    y_offset = (0.0, 10.0)

    openspace_width = 1.0
    sunlight_cut_start_height = 10.0
    sunlight_cut_ratio = 0.5
    floor_height = 3.0
    max_floor_count = 5

    law_settings = LawSettings(openspace_width, sunlight_cut_start_height, sunlight_cut_ratio, floor_height, max_floor_count)
    search_settings = SearchSapce(axis_index, pillar_x_width, pillar_y_width, x_intervals, y_intervals, x_offset, y_offset)
    env = ParcelEnv(parcel_data, search_settings, law_settings)

    env.initialize()

    for i in range(EPOCHS):
        # NOTE: 테스트용 random action
        action = search_settings.get_random_action()

        result = env.build(action)

        print(result)
