from itertools import chain
from collections import defaultdict
import cv2
from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta, BBox
from savant.utils.artist import Position, Artist
from samples.people_counting.utils import Direction, Movement, RandColorIterator


class Overlay(NvDsDrawFunc):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.obj_colors = defaultdict(lambda: next(RandColorIterator()))
        self.entry_text_anchor_pos = 0.1
        self.exit_text_anchor_pos = 0.2
        self.crowd_text_anchor_pos = 0.3
        self.idle_text_anchor_pos = 0.4

    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        line_to = None
        line_from = None
        entries_n = None
        exits_n = None
        is_crowded = None
        idles_n = None
        crowd_area = None
        interested_object_idxs = None

        for obj_meta in frame_meta.objects:
            if obj_meta.is_primary:
                line_from = obj_meta.get_attr_meta('analytics', 'line_from')
                line_to = obj_meta.get_attr_meta('analytics', 'line_to')
                entries_n = obj_meta.get_attr_meta('analytics', 'entries_n')
                exits_n = obj_meta.get_attr_meta('analytics', 'exits_n')
                is_crowded = obj_meta.get_attr_meta('crowd_analytics', 'is_crowded')
                crowd_area = obj_meta.get_attr_meta('crowd_analytics', 'crowd_area')
                idles_n = obj_meta.get_attr_meta('idle_analytics', 'idles_n')
                interested_object_idxs = obj_meta.get_attr_meta('interested_objects', 'object_idxs')
                interested_object_idxs = interested_object_idxs.value if interested_object_idxs is not None else []
                break

        for i, obj_meta in enumerate(frame_meta.objects):
            if (not obj_meta.is_primary
                and (interested_object_idxs is None or (interested_object_idxs and obj_meta.track_id in interested_object_idxs))
            ):
                # mark obj center as it is used for entry/exit detection
                color = self.obj_colors[(frame_meta.source_id, obj_meta.track_id)]
                artist.add_bbox(obj_meta.bbox, border_width=2, border_color=color)
                center = round(obj_meta.bbox.xc), round(obj_meta.bbox.yc)
                artist.add_circle(center, 3, color, cv2.FILLED)

                # add entry/exit label if detected
                entries = obj_meta.get_attr_meta_list(
                    'lc_tracker', Direction.entry.name
                )
                exits = obj_meta.get_attr_meta_list('lc_tracker', Direction.exit.name)
                entry_events_meta = entries if entries is not None else []
                exit_events_meta = exits if exits is not None else []
                offset = 20
                for attr_meta in chain(entry_events_meta, exit_events_meta):
                    direction = attr_meta.name
                    artist.add_text(
                        direction,
                        (int(obj_meta.bbox.left), int(obj_meta.bbox.top) + offset),
                        anchor_point_type=Position.LEFT_TOP,
                    )
                    offset += 20

                # add idle label if detected
                idle_objects = obj_meta.get_attr_meta(
                    'idle_tracker', Movement.idle.name
                )
                moving_objects = obj_meta.get_attr_meta(
                    'idle_tracker', Movement.moving.name
                )
                idle_events_meta = [idle_objects] if idle_objects is not None else []
                moving_events_meta = [moving_objects] if moving_objects is not None else []
                offset = 20
                for attr_meta in chain(idle_events_meta, moving_events_meta):
                    movement = attr_meta.name
                    artist.add_text(
                        movement,
                        (int(obj_meta.bbox.left), int(obj_meta.bbox.top) + offset),
                        anchor_point_type=Position.RIGHT_BOTTOM,
                    )
                    offset += 20

        # draw boundary lines
        if line_from and line_to:
            pt1 = line_from.value[:2]
            pt2 = line_from.value[2:]
            artist.add_polygon([pt1, pt2], line_color=(255, 0, 0, 255))
            pt1 = line_to.value[:2]
            pt2 = line_to.value[2:]
            artist.add_polygon([pt1, pt2], line_color=(0, 0, 255, 255))

        # manually refresh (by filling with black) frame padding used for drawing
        # this workaround avoids rendering problem where drawings from previous frames
        # are persisted on the padding area in the next frame
        frame_w, _ = artist.frame_wh
        artist.add_bbox(
            BBox(
                frame_w // 2,
                self.overlay_height // 2,
                frame_w,
                self.overlay_height,
            ),
            border_width=0,
            bg_color=(0, 0, 0, 0),
        )
        # add entries/exits counters
        entries_n = entries_n.value if entries_n is not None else 0
        exits_n = exits_n.value if exits_n is not None else 0
        artist.add_text(
            f'Entries: {entries_n}',
            (10, 30),
            0.5,
            2,
            anchor_point_type=Position.LEFT_TOP,
        )
        artist.add_text(
            f'Exits: {exits_n}',
            (10, 60),
            0.5,
            2,
            anchor_point_type=Position.LEFT_TOP,
        )

        # draw people crowding
        crowd_area = crowd_area.value if crowd_area is not None else []
        crowd_area = [crowd_area[i:i+2] for i in range(0, len(crowd_area), 2)]
        if crowd_area:
            artist.add_polygon(
                vertices=crowd_area,
                line_width=3,
                line_color=(255, 255, 255, 255)
            )
        crowd_text = 'yes' if is_crowded is not None and is_crowded.value else 'no'
        artist.add_text(
            f'Crowd detected: {crowd_text}',
            (10, 90),
            0.5,
            2,
            anchor_point_type=Position.LEFT_TOP,
        )

        # draw idle counts
        idles_n = idles_n.value if idles_n is not None else 0
        artist.add_text(
            f'# of standing people: {idles_n}',
            (150, 30),
            0.5,
            2,
            anchor_point_type=Position.LEFT_TOP
        )
