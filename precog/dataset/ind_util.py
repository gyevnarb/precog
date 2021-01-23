#  This file is a modified version of the config class from the file:
# https://github.com/nrhine1/precog/blob/master/precog/dataset/preprocess_nuscenes.py
from typing import List

import numpy as np
import cv2 as cv
import os

from lanelet2.core import Lanelet
from shapely.geometry import LineString


def collapse(arr):
    ret = arr[:, :, 0]
    for layer in arr.transpose((2, 0, 1)):
        mask = layer < 255
        ret[mask] = layer[mask]
    return ret


class InDConfig:
    def __init__(self, scenario, recordings_meta, draw_map = False):
        self.recordings_meta = recordings_meta
        self.draw_map = draw_map

        self._create_config()
        self.scenario_features = self._process_scenario(scenario)

    def _create_config(self):
        self.frame_rate = self.recordings_meta["frameRate"]  # Hz

        # samples are at 2Hz in the original Precog implementation
        self.sample_period = 2  # Hz
        self.sample_frequency = 1. / self.sample_period

        # Predict 4 seconds in the future with 1 second of past.
        self.past_horizon_seconds = 1  # Tp
        self.future_horizon_seconds = 2  # Tf

        # The number of samples we need.
        self.future_horizon_length = int(round(self.future_horizon_seconds * self.frame_rate))
        self.past_horizon_length = int(round(self.past_horizon_seconds * self.frame_rate))
        self.total_length = self.past_horizon_length + self.future_horizon_length

        #  Minimum OTHER agents visible. In our case all agents are OTHER agents
        # There is no dedicated ego vehicle
        self.min_relevant_agents = 1  # A

        # Image features to include in overhead features
        self.image_features = ["surfaces", "markings", "agents"]

        self.vehicle_colors = {
            "car": 0,
            "truck": 0,
            "truck_bus": 0
        }

        self.image_colors = {
            "background": 255,
            "walkway": 200,
            "exit": 180,
            "road": 63,
            "bicycle_lane": 127,
            "parking": 100,
            "freespace": 100,
            "vegetation": 100,
            "keepout": 100
        }

        # Space between dashed markings and length of dashes in metres
        self.dash_space = 2
        self.marking_length = 2
        self.marking_color = 240
        self.arrow_thickness = 2
        self.marking_thickness = 1

        #  Configuration for temporal interpolation.
        # For InD usually this is not necessary as the frame rate of the recording is high
        # but the sample rate is low
        self.target_sample_period_past = 5
        self.target_sample_period_future = 5
        self.target_sample_frequency_past = 1. / self.target_sample_period_past
        self.target_sample_frequency_future = 1. / self.target_sample_period_future
        self.target_past_horizon = self.past_horizon_seconds * self.target_sample_period_past
        self.target_future_horizon = self.future_horizon_seconds * self.target_sample_period_future
        self.target_past_times = -1 * np.arange(0, self.past_horizon_seconds, self.target_sample_frequency_past)[::-1]
        # Hacking negative zero to slightly positive for temporal interpolation purposes?
        self.target_past_times[np.isclose(self.target_past_times, 0.0, atol=1e-5)] = 1e-8

        # The final relative future times to interpolate
        self.target_future_times = np.arange(self.target_sample_frequency_future,
                                             self.future_horizon_seconds + self.target_sample_frequency_future,
                                             self.target_sample_frequency_future)

    def _process_scenario(self, scenario):
        features_list = []

        pavement_layer = self._get_surfaces_layer(scenario)
        features_list.extend(pavement_layer)

        markings_layer = self._get_markings_layer(scenario)
        features_list.extend(markings_layer)

        return features_list

    def _get_surfaces_layer(self, scenario):
        layer_keys = ["walkway", "exit", "road", "bicycle_lane", "parking"]
        w, h = scenario.display_wh
        layers = [np.full((h, w), self.image_colors["background"], np.uint8) for _ in range(len(layer_keys))]
        scale = 1.0 / scenario.config.background_px_to_meter

        def draw_surface(elem):
            if "subtype" in elem.attributes:
                polygon = elem.polygon2d() if isinstance(elem, Lanelet) else elem.outerBoundPolygon()
                points = [[int(scale * pt.x), int(-scale * pt.y)] for pt in polygon]
                if points and elem.attributes["subtype"] in layer_keys:
                    subtype = elem.attributes["subtype"]
                    layer_idx = layer_keys.index(subtype)
                    points = np.stack(points)
                    points[points < 0] = 0
                    points = points.reshape((-1, 1, 2))
                    cv.fillPoly(layers[layer_idx], [points], self.image_colors[subtype], cv.LINE_AA)

        for lanelet in scenario.lanelet_map.laneletLayer:
            draw_surface(lanelet)
        for area in scenario.lanelet_map.areaLayer:
            draw_surface(area)

        return layers

    def _get_markings_layer(self, scenario):
        marking_keys = ["line_thin_dashed", "line_thick_dashed", "line_thin_solid", "line_think_solid",
                        "arrow_straight", "arrow_left", "arrow_right", "arrow_straight_right", "arrow_straight_left"]
        w, h = scenario.display_wh
        layers = [np.full((h, w), self.image_colors["background"], np.uint8) for _ in range(2)]
        scale = 1.0 / scenario.config.background_px_to_meter

        for linestring in scenario.lanelet_map.lineStringLayer:
            if "type" in linestring.attributes and "subtype" in linestring.attributes:
                points = [[int(scale * pt.x), int(-scale * pt.y)] for pt in linestring]
                if not points:
                    continue

                points = np.stack(points)
                points[points < 0] = 0

                line_key = f"{linestring.attributes['type']}_{linestring.attributes['subtype']}"

                if line_key in marking_keys:
                    layer_idx = 0 if linestring.attributes["type"].startswith("line") else 1

                    if linestring.attributes["subtype"] == "dashed":
                        ls = LineString(points)
                        pts = []
                        for dist in np.arange(0, ls.length, scale * (self.dash_space + self.marking_length)):
                            start_pt = ls.interpolate(dist)
                            end_pt = ls.interpolate(dist + scale * self.marking_length)
                            line_pts = np.array([[start_pt.x, start_pt.y], [end_pt.x, end_pt.y]], np.int32)
                            line_pts = line_pts.reshape(-1, 1, 2)
                            pts.append(line_pts)
                        cv.polylines(layers[layer_idx], pts, False, self.marking_color, self.marking_thickness)

                    elif linestring.attributes["type"] == "arrow":
                        dir_vector = (points[1] - points[0]) / np.linalg.norm(points[1] - points[0])
                        norm_vector = np.array([[0, -1], [1, 0]]) @ dir_vector
                        if linestring.attributes["subtype"] in ["left", "right"]:
                            dir_mul = scale if linestring.attributes["subtype"] == "right" else -scale
                            p1 = points[1] + dir_mul * 0.7 * norm_vector
                            p2 = points[1] + dir_mul * 0.5 * norm_vector + scale * dir_vector
                            p3 = points[1] + dir_mul * 0.5 * norm_vector - scale * dir_vector
                            head = [p1, p2, p3, p1]

                        elif linestring.attributes["subtype"] in ["straight_left", "straight_right"]:
                            dir_mul = scale if linestring.attributes["subtype"] == "straight_right" else -scale
                            p0 = points[1] - scale * 2 * dir_vector
                            p1 = p0 + dir_mul * 0.7 * norm_vector
                            p2 = p0 + dir_mul * 0.5 * norm_vector + scale * 0.75 * dir_vector
                            p3 = p0 + dir_mul * 0.5 * norm_vector - scale * 0.75 * dir_vector
                            p4 = points[1] + scale * 2 * dir_vector
                            p5 = points[1] + scale * 0.3 * norm_vector
                            p6 = points[1] - scale * 0.3 * norm_vector
                            head = [p0, p1, p2, p3, p1, p0, p4, p5, p6, p4]

                        else:  # Straight arrow
                            p1 = points[1] - scale * 2 * dir_vector + scale * 0.3 * norm_vector
                            p2 = points[1] - scale * 2 * dir_vector - scale * 0.3 * norm_vector
                            head = [p1, p2, points[1]]

                        points = np.append(points, head, axis=0).astype(np.int32)
                        points = points.reshape(-1, 1, 2)
                        cv.polylines(layers[layer_idx], [points], False, self.marking_color, self.arrow_thickness)

                    else:
                        points = points.reshape(-1, 1, 2)
                        cv.polylines(layers[layer_idx], [points], False, self.marking_color, self.marking_thickness)
        return layers


class InDMultiagentDatum:
    t = 0

    def __init__(self, player_past, agent_pasts, player_future, agent_futures,
                 player_yaw, agent_yaws, overhead_features, metadata={}):
        self.player_past = player_past
        self.agent_pasts = agent_pasts
        self.player_future = player_future
        self.agent_futures = agent_futures
        self.player_yaw = player_yaw
        self.agent_yaws = agent_yaws
        self.overhead_features = overhead_features
        self.metadata = metadata

    @classmethod
    def from_ind_trajectory(cls, agents_to_include: List[int], episode, scenario,
                            reference_frame: int, cfg: InDConfig):
        # There is no explicit ego in the InD dataset
        player_past = []
        player_future = []
        player_yaw = None

        all_agent_pasts = []
        all_agent_futures = []
        all_agent_yaws = []

        agent_dims = []

        for agent_id in agents_to_include:
            agent = episode.agents[agent_id]
            local_frame = reference_frame - agent.initial_frame

            agent_past_trajectory = agent.state_history[local_frame - cfg.past_horizon_length:local_frame]
            all_agent_pasts.append([[frame.x, frame.y, 0.0] for frame in agent_past_trajectory])

            agent_future_trajectory = agent.state_history[local_frame:local_frame + cfg.future_horizon_length]
            all_agent_futures.append([[frame.x, frame.y, 0.0] for frame in agent_future_trajectory])

            all_agent_yaws.append(agent.state_history[local_frame].heading)

            agent_dims.append([agent.width, agent.length, agent.agent_type])

        # TODO: Use UTM coordinate frame instead?
        # (A, Tp, d). Agent past trajectories in local frame at t=now
        agent_pasts = np.stack(all_agent_pasts, axis=0)
        # (A, Tf, d). Agent future trajectories in local frame at t=now
        agent_futures = np.stack(all_agent_futures, axis=0)
        # (A,). Agent yaws in ego frame at t=now
        agent_yaws = np.asarray(all_agent_yaws)

        assert agent_pasts.shape[0] == agent_futures.shape[0] == agent_yaws.shape[0] == len(agents_to_include)

        overhead_features = InDMultiagentDatum.get_image_features(
            scenario, agent_pasts[:, -1, :], agent_yaws, agent_dims, cfg)

        metadata = {}

        return cls(player_past, agent_pasts,
                   player_future, agent_futures,
                   player_yaw, agent_yaws,
                   overhead_features,
                   metadata)

    @classmethod
    def get_image_features(cls, scenario, agent_poses, agent_yaws, agent_dims, cfg):
        features_list = cfg.scenario_features.copy()
        agents_layer = InDMultiagentDatum.get_agent_boxes(scenario, agent_poses, agent_yaws, agent_dims, cfg)
        features_list.append(agents_layer)

        image = np.stack(features_list, axis=-1)
        image = image[:708, :905, :]  # Trim image to smallest of all scenarios

        if cfg.draw_map:
            path_to_viz = f"precog/{scenario.config.name}/map_viz/"
            if not os.path.exists(path_to_viz):
                os.mkdir(path_to_viz)
            cv.imwrite(os.path.join(path_to_viz, f"{cls.t}.png"), collapse(image))
            cls.t += 1

        # Turn image to "binary"
        mask = image != 255
        image[mask] = 0

        # for i, l in enumerate(image.transpose((-1, 0, 1))):
        #     cv.imwrite(get_data_dir() + f"precog/{i}_feature.png", l)

        return image

    @staticmethod
    def get_agent_boxes(scenario, agent_poses, agent_yaws, agent_dims, cfg):
        w, h = scenario.display_wh
        layer = np.full((h, w), cfg.image_colors["background"], np.uint8)

        for i in range(agent_poses.shape[0]):
            agent_dim = agent_dims[i]
            agent_box = Box(agent_poses[i, :], agent_dim[0], agent_dim[1], agent_yaws[i]).get_display_box(scenario)
            agent_box = agent_box.reshape((-1, 1, 2))
            color = cfg.vehicle_colors[agent_dim[2]]
            cv.fillPoly(layer, [agent_box], color, cv.LINE_AA)
        return layer


class Box:
    def __init__(self, center, width, length, heading):
        self.center = np.array(center)[:2]
        self.width = width
        self.length = length
        self.heading = heading

        self.bounding_box = self.get_bounding_box()

    def get_bounding_box(self):
        box = np.array(
            [np.array([-self.length / 2, self.width / 2]),  # top-left
             np.array([self.length / 2, self.width / 2]),  # top-right
             np.array([-self.length / 2, -self.width / 2]),  # bottom-left
             np.array([self.length / 2, -self.width / 2])]  # bottom-right
        )

        rotation = np.array([[np.cos(self.heading), -np.sin(self.heading)],
                             [np.sin(self.heading), np.cos(self.heading)]])
        return self.center + box @ rotation.T

    def get_display_box(self, scenario):
        scale = 1.0 / scenario.config.background_px_to_meter
        permuter = np.array(  # Need to permute vertices for display
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1],
             [0, 0, 1, 0]]
        )

        return permuter @ (scale * np.abs(self.bounding_box)).astype(np.int32)
