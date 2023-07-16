from collections import defaultdict
import sys
import time
import yaml
import numpy as np
from savant_rs.primitives.geometry import PolygonalArea, Point
from savant.gstreamer import Gst
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from samples.traffic_meter.utils import (
    Point, Direction, TwoLinesCrossingTracker,
    IdleObjectTracker, CrowdTracker, SpeedEstimator, Movement,
    is_inside_postgis_parallel
)


class ConditionalDetectorSkip(NvDsPyFuncPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with open(self.config_path, "r", encoding='utf8') as stream:
            self.line_config = yaml.safe_load(stream)

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        primary_meta_object = None
        for obj_meta in frame_meta.objects:
            if obj_meta.is_primary:
                primary_meta_object = obj_meta
                break

        # if the boundary lines are not configured for this source
        # then disable detector inference entirely by removing the primary object
        # Note:
        # In order to enable use cases such as conditional inference skip
        # or user-defined ROI, Savant configures all Deepstream models to run
        # in 'secondary' mode and inserts a primary 'frame' object into the DS meta
        if (
            primary_meta_object is not None
            and frame_meta.source_id not in self.line_config
        ):
            frame_meta.remove_obj_meta(primary_meta_object)


class LineCrossing(NvDsPyFuncPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with open(self.config_path, "r", encoding='utf8') as stream:
            self.line_config = yaml.safe_load(stream)

        self.areas = {}
        for source_id, line_cfg in self.line_config.items():
            # The conversion from 2 lines to a 4 point polygon is as follows:
            # assuming the lines are AB and CD, the polygon is ABDC.
            # The AB polygon edge is marked as "from" and the CD edge is marked as "to".
            pt_A = Point(*line_cfg['from'][:2])
            pt_B = Point(*line_cfg['from'][2:])
            pt_C = Point(*line_cfg['to'][:2])
            pt_D = Point(*line_cfg['to'][2:])
            area = PolygonalArea([pt_A, pt_B, pt_D, pt_C], ["from", None, "to", None])
            if area.is_self_intersecting():
                # try to correct the polygon by reversing one of the lines
                area = PolygonalArea(
                    [pt_A, pt_B, pt_C, pt_D], ["from", None, "to", None]
                )
                if area.is_self_intersecting():
                    self.logger.error(
                        'Lines config for the "%s" source id produced a self-intersecting polygon.'
                        ' Please correct coordinates "%s" in the config file and restart the pipeline.',
                        source_id,
                        line_cfg,
                    )
                    sys.exit(1)
            self.areas[source_id] = area

        # load calibration data
        self.calib_config = {}
        with open(self.calib_path, 'r', encoding='utf8') as f:
            self.calib_data = yaml.safe_load(f)
        for source_id, calibration_data in self.calib_data.items():
            self.calib_config[source_id] = calibration_data

        self.lc_trackers = {}
        self.track_last_frame_num = defaultdict(lambda: defaultdict(int))
        self.entry_count = defaultdict(int)
        self.exit_count = defaultdict(int)
        self.cross_events = defaultdict(lambda: defaultdict(list))

        self.idle_trackers = {}
        self.crowd_trackers = {}
        self.speed_estimators = {}

    def on_source_eos(self, source_id: str):
        """On source EOS event callback."""
        if source_id in self.lc_trackers:
            del self.lc_trackers[source_id]
        if source_id in self.track_last_frame_num:
            del self.track_last_frame_num[source_id]
        if source_id in self.cross_events:
            del self.cross_events[source_id]
        if source_id in self.entry_count:
            del self.entry_count[source_id]
        if source_id in self.exit_count:
            del self.exit_count[source_id]

        if source_id in self.idle_trackers:
            del self.idle_trackers[source_id]
        if source_id in self.crowd_trackers:
            del self.crowd_trackers[source_id]
        if source_id in self.speed_estimators:
            del self.speed_estimators[source_id]

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Process frame metadata.

        :param buffer: Gstreamer buffer with this frame's data.
        :param frame_meta: This frame's metadata.
        """
        # the primary meta object may be missed in the first several frames
        # due to nvtracker deleting all unconfirmed tracks
        primary_meta_object = None
        for obj_meta in frame_meta.objects:
            if obj_meta.is_primary:
                primary_meta_object = obj_meta
                break

        if primary_meta_object is not None and frame_meta.source_id in self.areas:
            if frame_meta.source_id not in self.lc_trackers:
                self.lc_trackers[frame_meta.source_id] = TwoLinesCrossingTracker(
                    self.areas[frame_meta.source_id]
                )

            lc_tracker = self.lc_trackers[frame_meta.source_id]

            if frame_meta.source_id not in self.idle_trackers:
                self.idle_trackers[frame_meta.source_id] = IdleObjectTracker(
                    idle_tracker_buffer=self.idle_tracker_buffer,
                    idle_distance_threshold=self.idle_distance_threshold
                )
            idle_trakcer = self.idle_trackers[frame_meta.source_id]

            crowd_area = self.calib_config[frame_meta.source_id]['crowd']['crowd_area']
            crowd_threshold = self.calib_config[frame_meta.source_id]['crowd']['crowd_threshold']
            if frame_meta.source_id not in self.crowd_trackers:
                self.crowd_trackers[frame_meta.source_id] = CrowdTracker(
                    crowd_area=crowd_area
                )
            crowd_tracker = self.crowd_trackers[frame_meta.source_id]

            if frame_meta.source_id not in self.speed_estimators:
                self.speed_estimators[frame_meta.source_id] = SpeedEstimator(
                    self.calib_config[frame_meta.source_id]['speed']['speed_area'],
                    self.calib_config[frame_meta.source_id]['speed']['speed_area_real_width'],
                    self.calib_config[frame_meta.source_id]['speed']['speed_area_real_height'],
                )
            speed_estimator = self.speed_estimators[frame_meta.source_id]

            crowd_area_2d = np.array(crowd_area, dtype=np.float32).reshape((-1, 2))
            obj_metas = []
            interested_object_idxs = []
            for i, obj_meta in enumerate(frame_meta.objects):
                if obj_meta.label in self.target_obj_labels:
                    # keep objects that are inside the RoI only
                    centroid = np.array([[obj_meta.bbox.xc, obj_meta.bbox.yc]], dtype=np.float32)
                    if np.all(is_inside_postgis_parallel(centroid, crowd_area_2d)):
                        lc_tracker.add_track_point(
                            obj_meta.track_id,
                            # center point
                            Point(
                                obj_meta.bbox.xc,
                                obj_meta.bbox.yc,
                            ),
                        )
                        object_coordinate = (obj_meta.bbox.xc, obj_meta.bbox.yc)
                        idle_trakcer.update(obj_meta.track_id, object_coordinate)
                        crowd_tracker.update(object_coordinate)
                        speed_estimator.update(obj_meta.track_id, object_coordinate, time.time())

                        self.track_last_frame_num[frame_meta.source_id][
                            obj_meta.track_id
                        ] = frame_meta.frame_num

                        obj_metas.append(obj_meta)
                        interested_object_idxs.append(i)

            idles_count = 0
            if obj_metas:
                track_lines_crossings = lc_tracker.check_tracks(
                    [obj_meta.track_id for obj_meta in obj_metas]
                )

                speeds = speed_estimator.estimate(
                    [obj_meta.track_id for obj_meta in obj_metas]
                )

                idle_objects = idle_trakcer.check_idle(
                    [obj_meta.track_id for obj_meta in obj_metas]
                )
                for obj_meta, cross_direction, speed, movement in zip(obj_metas, track_lines_crossings, speeds, idle_objects):
                    obj_events = self.cross_events[frame_meta.source_id][obj_meta.track_id]
                    if cross_direction is not None:

                        obj_events.append((cross_direction.name, frame_meta.pts))

                        if cross_direction == Direction.entry:
                            self.entry_count[frame_meta.source_id] += 1
                        elif cross_direction == Direction.exit:
                            self.exit_count[frame_meta.source_id] += 1

                    for direction_name, frame_pts in obj_events:
                        obj_meta.add_attr_meta('lc_tracker', direction_name, frame_pts)

                    obj_meta.add_attr_meta('speed_tracker', 'speed', speed)

                    obj_meta.add_attr_meta('idle_tracker', movement, frame_meta.pts)
                    if movement == Movement.idle.name:
                        idles_count += 1
                
            # interested objects
            primary_meta_object.add_attr_meta(
                'interested_objects', 'object_idxs', interested_object_idxs
            )

            # crowd detection
            is_crowded = crowd_tracker.check_crowd(crowd_threshold)
            primary_meta_object.add_attr_meta(
                'crowd_analytics', 'crowd_area', crowd_area
            )
            primary_meta_object.add_attr_meta(
                'crowd_analytics', 'is_crowded', 1 if is_crowded else 0
            )

            # idle detection
            primary_meta_object.add_attr_meta(
                'idle_analytics', 'idles_n', idles_count
            )

            primary_meta_object.add_attr_meta(
                'analytics', 'entries_n', self.entry_count[frame_meta.source_id]
            )
            primary_meta_object.add_attr_meta(
                'analytics', 'exits_n', self.exit_count[frame_meta.source_id]
            )
            primary_meta_object.add_attr_meta(
                'analytics', 'line_from', self.line_config[frame_meta.source_id]['from']
            )
            primary_meta_object.add_attr_meta(
                'analytics', 'line_to', self.line_config[frame_meta.source_id]['to']
            )

        # periodically remove stale tracks
        if not (frame_meta.frame_num % self.stale_track_del_period):
            last_frames = self.track_last_frame_num[frame_meta.source_id]

            to_delete = [
                track_id
                for track_id, last_frame in last_frames.items()
                if frame_meta.frame_num - last_frame > self.stale_track_del_period
            ]
            if to_delete:
                for track_id in to_delete:
                    lc_tracker = self.lc_trackers[frame_meta.source_id]
                    del last_frames[track_id]
                    lc_tracker.remove_track(track_id)