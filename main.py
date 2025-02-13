from shapely.io import from_wkt

import debugvisualizer as dv  # pylint: disable=unused-import
from domino_gym.env import ParcelEnv, LawSettings, SearchSapce, BuildResult

PARCEL_EXAMPLES = [
    "POLYGON ((0 0, 30 0, 20 20, 0 10, 0 0))",
    "POLYGON((195859.702559397701 448783.861942475953,195860.105651073245 448771.277448283101,195869.311817773851 448771.550985747599,195878.583749606594 448771.799834780279,195879.552729209652 448771.825720670633,195885.499756585341 448771.971523154411,195885.222415796394 448777.467704985582,195880.193718167488 448779.841820573609,195879.166974653112 448780.348130546685,195878.150551098865 448780.844221398409,195876.687028007902 448781.643144439673,195871.343772268447 448784.232874869369,195868.774261611921 448784.180405020306,195859.702559397701 448783.861942475953))",
]

TRIALS = 1000

if __name__ == "__main__":
    parcel_polygon = from_wkt(PARCEL_EXAMPLES[1])

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
    x_rows_max = 10
    y_rows_max = 10

    law_settings = LawSettings(openspace_width, sunlight_cut_start_height, sunlight_cut_ratio, floor_height, max_floor_count)
    search_settings = SearchSapce(axis_index, pillar_x_width, pillar_y_width, x_intervals, y_intervals, x_offset, y_offset, x_rows_max, y_rows_max)
    env = ParcelEnv(parcel_polygon, search_settings, law_settings)

    env.initialize()

    best_result: BuildResult = None
    for i in range(TRIALS):
        # NOTE: 테스트용 random action
        action = search_settings.get_random_action()

        build_result: BuildResult = env.build(action)

        # NOTE: using total area as a temporary score
        if best_result is None or build_result.total_area > best_result.total_area:
            print(f"New best result found - Trial {i}: {build_result.total_area}")
            best_result = build_result

        if i % 100 == 0:
            print(f"Trial {i}: {best_result.total_area}")

    print(f"Final best result: {best_result.total_area}")
