#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-05-27 15:00
# @Author  : Xiaoke Huang
# @Email   : xiaokehuang@foxmail.com
import pdb
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os
from utils.config import LANE_WIDTH


def get_halluc_lane(centerlane, city_name):
    """
    return left & right lane based on centerline
    args:
    returns:
        doubled_left_halluc_lane, doubled_right_halluc_lane, shaped in (N-1, 3)
    """
    if centerlane.shape[0] <= 1:
        raise ValueError('shape of centerlane error.')

    half_width = LANE_WIDTH[city_name] / 2
    rotate_quat = np.array([[0.0, -1.0], [1.0, 0.0]])
    halluc_lane_1, halluc_lane_2 = np.empty(
        (0, centerlane.shape[1]*2)), np.empty((0, centerlane.shape[1]*2))
    for i in range(centerlane.shape[0]-1):
        st, en = centerlane[i][:2], centerlane[i+1][:2]
        dx = en - st
        norm = np.linalg.norm(dx)
        e1, e2 = rotate_quat @ dx / norm, rotate_quat.T @ dx / norm
        lane_1 = np.hstack(
            (st + e1 * half_width, centerlane[i][2], en + e1 * half_width, centerlane[i+1][2]))
        lane_2 = np.hstack(
            (st + e2 * half_width, centerlane[i][2], en + e2 * half_width, centerlane[i+1][2]))
        # print(halluc_lane_1, )
        halluc_lane_1 = np.vstack((halluc_lane_1, lane_1))
        halluc_lane_2 = np.vstack((halluc_lane_2, lane_2))
    return halluc_lane_1, halluc_lane_2


def get_rect_lane_id(lane_dict, x_min, x_max, y_min, y_max):
    lane_ids = []

    # Get lane centerlines which lie within the range of trajectories
    for lane_id, lane_props in lane_dict.items():

        lane_cl = lane_props.centerline

        if (
            np.min(lane_cl[:, 0]) < x_max
            and np.min(lane_cl[:, 1]) < y_max
            and np.max(lane_cl[:, 0]) > x_min
            and np.max(lane_cl[:, 1]) > y_min
        ):
            lane_ids.append(lane_id)
    return lane_ids


def get_nearby_lane_feature_ls(am, agent_df, obs_len, city_name, lane_radius, norm_center, has_attr=False, mode='nearby', query_bbox=None):
    '''
    compute lane features
    args:
        norm_center: np.ndarray
        mode: 'nearby' return nearby lanes within the radius; 'rect' return lanes within the query bbox
        **kwargs: query_bbox= List[int, int, int, int]
    returns:
        list of list of lane a segment feature, formatted in [left_lane, right_lane, is_traffic_control, is_intersection, lane_id]
    '''
    # 道路特征列表
    lane_feature_ls = []
    if mode == 'nearby':
        # 当前自车的位置
        query_x, query_y = agent_df[['X', 'Y']].values[obs_len-1]
        # 以自车位置和城市为索引，获得半径范围内所有的道路编号，圆形
        nearby_lane_ids = am.get_lane_ids_in_xy_bbox(
            query_x, query_y, city_name, lane_radius)

        for lane_id in nearby_lane_ids:
            # 某个车道是否具有交通控制措施的一个特定属性或字段。这可能包括了交通信号灯、标线、交叉口规划等信息
            # true / false
            traffic_control = am.lane_has_traffic_control_measure(
                lane_id, city_name)
            # 描述车道是否位于交叉口内
            is_intersection = am.lane_is_in_intersection(lane_id, city_name)
            
            # 获得道路中心线, <class 'numpy.ndarray'> shape: n * 3
            centerlane = am.get_lane_segment_centerline(lane_id, city_name)
            
            # normalize to last observed timestamp point of agent
            # 将所有的车道中心线的坐标转换成为相对于当前自车的坐标
            centerlane[:, :2] -= norm_center
            # 获得道路两边的线的坐标
            halluc_lane_1, halluc_lane_2 = get_halluc_lane(
                centerlane, city_name)

            # pdb.set_trace()
            # BUG: halluc_lane_1, halluc_lane_2 might contain nan, throwing to upper call function to handle
            # (present solution, make z (hight) all be average)
            # if numpy.isnan(halluc_lane_1[:, 2]).any():
            # halluc_lane_1[:, 2].fill(.0)
            # halluc_lane_1[:, 5].fill(.0)
            # halluc_lane_2[:, 2].fill(.0)
            # halluc_lane_2[:, 5].fill(.0)

            if has_attr:
                raise NotImplementedError()

            lane_feature_ls.append(
                [halluc_lane_1, halluc_lane_2, traffic_control, is_intersection, lane_id])
    elif mode == 'rect':
        lane_dict = am.city_lane_centerlines_dict[city_name]
        query_x, query_y = agent_df[['X', 'Y']].values[obs_len-1]
        # nearby_lane_ids = am.get_lane_ids_in_xy_bbox(query_x, query_y, city_name, lane_radius)
        nearby_lane_ids = get_rect_lane_id(
            lane_dict, query_bbox[0]+query_x, query_bbox[1]+query_x, query_bbox[2]+query_y, query_bbox[3]+query_y)

        for lane_id in nearby_lane_ids:
            traffic_control = am.lane_has_traffic_control_measure(
                lane_id, city_name)
            is_intersection = am.lane_is_in_intersection(lane_id, city_name)

            centerlane = am.get_lane_segment_centerline(lane_id, city_name)
            # normalize to last observed timestamp point of agent
            centerlane[:, :2] -= norm_center
            halluc_lane_1, halluc_lane_2 = get_halluc_lane(
                centerlane, city_name)

            # pdb.set_trace()
            # BUG: halluc_lane_1, halluc_lane_2 might contain nan, throwing to upper call function to handle
            # (present solution, make z (hight) all be average)
            # if numpy.isnan(halluc_lane_1[:, 2]).any():
            # halluc_lane_1[:, 2].fill(.0)
            # halluc_lane_1[:, 5].fill(.0)
            # halluc_lane_2[:, 2].fill(.0)
            # halluc_lane_2[:, 5].fill(.0)

            if has_attr:
                raise NotImplementedError()

            lane_feature_ls.append(
                [halluc_lane_1, halluc_lane_2, traffic_control, is_intersection, lane_id])
    else:
        raise ValueError(f"{mode} is not in {'rect', 'nearby'}")

    return lane_feature_ls
    # polygon = am.get_lane_segment_polygon(lane_id, city_name)
    # h_len = polygon.shape[0]
    # polygon = np.hstack(
    #     (polygon, is_intersection * np.ones((h_len, 1)), traffic_control * np.ones((h_len, 1))))
    # polygon_ls.append(polygon)
