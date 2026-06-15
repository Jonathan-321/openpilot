#!/usr/bin/env python3
import os
os.environ['GMMU'] = '0'  # for usbgpu fast loading, noop for qcom
import time
import numpy as np
import cereal.messaging as messaging
from cereal import car, log
from cereal.messaging import SubMaster
from msgq.visionipc import VisionIpcClient, VisionStreamType
from opendbc.car.car_helpers import get_demo_car_params
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import config_realtime_process, DT_MDL
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.modeld import ModelState, FrameMeta, LAT_SMOOTH_SECONDS, LONG_SMOOTH_SECONDS
from openpilot.selfdrive.modeld.model_channel import ModelChannel


def run(usbgpu: bool, channel_path: str, core, priority: int = 53, demo=False):
  name = "bigmodeld" if usbgpu else "smallmodeld"
  cloudlog.warning(f"{name} init")
  config_realtime_process(core, priority)
  params = Params()
  channel = ModelChannel(channel_path, create=True)

  cloudlog.warning(f"{name} waiting for camerad vision")
  while True:
    available_streams = VisionIpcClient.available_streams("camerad", block=False)
    if available_streams:
      use_extra_client = VisionStreamType.VISION_STREAM_WIDE_ROAD in available_streams and VisionStreamType.VISION_STREAM_ROAD in available_streams
      main_wide_camera = VisionStreamType.VISION_STREAM_ROAD not in available_streams
      break
    time.sleep(.1)

  vipc_client_main_stream = VisionStreamType.VISION_STREAM_WIDE_ROAD if main_wide_camera else VisionStreamType.VISION_STREAM_ROAD
  vipc_client_main = VisionIpcClient("camerad", vipc_client_main_stream, True)
  vipc_client_extra = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, False)
  while not vipc_client_main.connect(False):
    time.sleep(0.1)
  while use_extra_client and not vipc_client_extra.connect(False):
    time.sleep(0.1)
  cloudlog.warning(f"{name} vision connected")

  st = time.monotonic()
  cloudlog.warning(f"{name} loading model")
  try:
    model = ModelState(vipc_client_main.width, vipc_client_main.height, usbgpu)
  except Exception:
    # usbgpu load failed. mark failed so the UI shows SMALL not LOADING, and park instead of crashing
    # (avoids a "not running" alert). the selector keeps publishing small until the next ignition.
    cloudlog.exception(f"{name} model load failed, parking until next ignition cycle")
    while True:
      time.sleep(1)
  cloudlog.warning(f"{name} loaded model in {time.monotonic() - st:.1f}s")

  sm = SubMaster(["deviceState", "carState", "roadCameraState", "liveCalibration", "driverMonitoringState", "carControl", "liveDelay"])
  if demo:
    CP = get_demo_car_params()
  else:
    CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  long_delay = CP.longitudinalActuatorDelay + LONG_SMOOTH_SECONDS
  DH = DesireHelper()
  frame_dropped_filter = FirstOrderFilter(0., 10., 1. / ModelConstants.MODEL_RUN_FREQ)

  model_transform_main = np.zeros((3, 3), dtype=np.float32)
  model_transform_extra = np.zeros((3, 3), dtype=np.float32)
  live_calib_seen = False
  buf_main, buf_extra = None, None
  meta_main = FrameMeta()
  meta_extra = FrameMeta()
  last_vipc_frame_id = 0
  run_count = 0
  produced_count = 0

  while True:
    while meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
      buf_main = vipc_client_main.recv()
      meta_main = FrameMeta(vipc_client_main)
      if buf_main is None:
        break
    if buf_main is None:
      continue

    if use_extra_client:
      while True:
        buf_extra = vipc_client_extra.recv()
        meta_extra = FrameMeta(vipc_client_extra)
        if buf_extra is None or meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
          break
      if buf_extra is None:
        continue
    else:
      buf_extra = buf_main
      meta_extra = meta_main

    sm.update(0)
    desire = DH.desire
    is_rhd = sm["driverMonitoringState"].isRHD
    frame_id = sm["roadCameraState"].frameId
    v_ego = max(sm["carState"].vEgo, 0.)
    lat_delay = sm["liveDelay"].lateralDelay + LAT_SMOOTH_SECONDS
    if sm.updated["liveCalibration"] and sm.seen['roadCameraState'] and sm.seen['deviceState']:
      device_from_calib_euler = np.array(sm["liveCalibration"].rpyCalib, dtype=np.float32)
      dc = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['roadCameraState'].sensor))]
      model_transform_main = get_warp_matrix(device_from_calib_euler, dc.ecam.intrinsics if main_wide_camera else dc.fcam.intrinsics, False).astype(np.float32)
      model_transform_extra = get_warp_matrix(device_from_calib_euler, dc.ecam.intrinsics, True).astype(np.float32)
      live_calib_seen = True

    traffic_convention = np.zeros(2)
    traffic_convention[int(is_rhd)] = 1
    vec_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
    if desire >= 0 and desire < ModelConstants.DESIRE_LEN:
      vec_desire[desire] = 1

    vipc_dropped_frames = max(0, meta_main.frame_id - last_vipc_frame_id - 1)
    frames_dropped = frame_dropped_filter.update(min(vipc_dropped_frames, 10))
    if run_count < 10:
      frame_dropped_filter.x = 0.
      frames_dropped = 0.
    run_count += 1
    frame_drop_ratio = frames_dropped / (1 + frames_dropped)
    prepare_only = vipc_dropped_frames > 0

    bufs = {n: buf_extra if 'big' in n else buf_main for n in model.vision_input_names}
    transforms = {n: model_transform_extra if 'big' in n else model_transform_main for n in model.vision_input_names}
    frame_delay = DT_MDL
    action_delay = DT_MDL / 2
    lat_action_t = lat_delay + frame_delay + action_delay
    long_action_t = long_delay + frame_delay + action_delay
    inputs = {
      'desire_pulse': vec_desire,
      'traffic_convention': traffic_convention,
      'action_t': np.array([lat_action_t, long_action_t], dtype=np.float32),
    }

    mt1 = time.perf_counter()
    try:
      model_output = model.run(bufs, transforms, inputs, prepare_only)
    except Exception:
      # usbgpu errored/disconnected. modeld is already on small with no gap. idle instead of exiting
      # (exiting trips "bigmodeld not running") and don't touch the usbgpu again until next ignition
      cloudlog.exception(f"{name} model run failed, parking until next ignition cycle")
      while True:
        time.sleep(1)
    model_execution_time = time.perf_counter() - mt1
    if model_output is not None:
      desire_state = model_output['desire_state'][0].reshape(-1)
      lane_change_prob = desire_state[log.Desire.laneChangeLeft] + desire_state[log.Desire.laneChangeRight]
      DH.update(sm['carState'], sm['carControl'].latActive, lane_change_prob)
      payload = {
        'output': model_output,
        'frame_id': meta_main.frame_id,
        'frame_id_extra': meta_extra.frame_id,
        'road_frame_id': frame_id,
        'timestamp_eof': meta_main.timestamp_eof,
        'vipc_dropped_frames': vipc_dropped_frames,
        'frame_drop_ratio': frame_drop_ratio,
        'live_calib_seen': live_calib_seen,
        'lat_action_t': lat_action_t,
        'long_action_t': long_action_t,
        'v_ego': v_ego,
        'model_execution_time': model_execution_time,
        'lane_change_state': int(DH.lane_change_state),
        'lane_change_direction': int(DH.lane_change_direction),
      }
      channel.write(meta_main.frame_id, payload)
      produced_count += 1
      # heartbeat in the rlog so we see the worker is producing and how fast. a worker that loads but
      # never produces shows a "loaded model" line and no "producing" line
      if produced_count == 1 or produced_count % 100 == 0:
        cloudlog.warning(f"{name} producing: frame={meta_main.frame_id} exec={model_execution_time * 1e3:.0f}ms "
                         f"dropped={vipc_dropped_frames} count={produced_count}")
    last_vipc_frame_id = meta_main.frame_id
