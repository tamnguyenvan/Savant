"""Config builder for Gst-nvstreammux element."""
from fractions import Fraction


def build_nvstreammux_config(
    batch_size: int = 1,
    adaptive_batching: bool = True,
    max_fps_control: bool = False,
    max_same_source_frames: int = 1,
    overall_min_fps: Fraction = Fraction(5, 1),
    overall_max_fps: Fraction = Fraction(120, 1),
    algorithm_type: int = 1,
) -> str:
    """Build config for Gst-nvstreammux element.

    See https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvstreammux2.html#mux-config-properties
    for details.

    :param batch_size: The desired batch size.
    :param adaptive_batching: Enable or disable adaptive batching.
        If enabled, batch-size is == number of sources X num-surfaces-per-frame.
    :param max_fps_control: Enable or disable controlling the maximum frame-rate at which nvstreammux pushes out
        batch buffers based on the overall_max_fps configuration.
    :param max_same_source_frames: Max number of any stream’s frames allowed to be muxed per output batch buffer.
    :param overall_min_fps: Desired overall muxer output min frame rate
    :param overall_max_fps: Desired overall muxer output max frame rate.
        Note: This value needs to be configured to a value >= overall_min_fps even when max_fps_control is disabled.
    :param algorithm_type: Defines the batching algorithm.
        If 1: Round-robbin if all sources have same priority key setting.
        Otherwise higher priority streams will be batched until no more buffers from them.
    """

    properties = {
        'batch-size': batch_size,
        'adaptive-batching': int(adaptive_batching),
        'max-fps-control': int(max_fps_control),
        'max-same-source-frames': max_same_source_frames,
        'overall-max-fps-n': overall_max_fps.numerator,
        'overall-max-fps-d': overall_max_fps.denominator,
        'overall-min-fps-n': overall_min_fps.numerator,
        'overall-min-fps-d': overall_min_fps.denominator,
        'algorithm-type': algorithm_type,
    }
    lines = ['[property]']
    for k, v in properties.items():
        lines.append(f'{k}={v}')
    config_str = '\n'.join(lines)

    return config_str
