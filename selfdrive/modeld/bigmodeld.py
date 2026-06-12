#!/usr/bin/env python3
import os
os.environ['GMMU'] = '0'  # for usbgpu fast loading
import time
import numpy as np
import cereal.messaging as messaging
from cereal import car, log
from cereal.messaging import SubMaster
from msgq.visionipc import VisionIpcClient, VisionStreamType
from opendbc.car.car_helpers import get_demo_car_params
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, DT_MDL
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.common.file_chunker import get_manifest_path
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.helpers import usbgpu_present, modeld_pkl_path
from openpilot.selfdrive.modeld.modeld import ModelState, FrameMeta, LAT_SMOOTH_SECONDS, LONG_SMOOTH_SECONDS
from openpilot.selfdrive.modeld.big_model_ipc import BigModelChannel

PROCESS_NAME = "selfdrive.modeld.bigmodeld"
USBGPU_TRANSFER_TIMEOUT_MS = 100


def cap_usbgpu_transfer_timeouts(max_ms: int = USBGPU_TRANSFER_TIMEOUT_MS) -> None:
  try:
    from tinygrad.runtime.support import usb as tgusb
  except Exception:
    cloudlog.exception("could not import tinygrad usb support to cap timeouts")
    return
  for name in ("_bulk_in", "_bulk_out"):
    orig = getattr(tgusb.USB3, name)
    def make_capped(orig_fn):
      def capped(self, ep, arg, timeout=1000):
        return orig_fn(self, ep, arg, min(timeout, max_ms))
      return capped
    setattr(tgusb.USB3, name, make_capped(orig))


# runs the big model on the usbgpu in its own process and writes each frame to modeld via shared memory
def main(demo=False):
  cloudlog.warning("bigmodeld init")
  if not (usbgpu_present() and os.path.isfile(get_manifest_path(modeld_pkl_path(usbgpu=True)))):
    cloudlog.warning("no usbgpu / big model, bigmodeld exiting")
    return

  cap_usbgpu_transfer_timeouts()
  config_realtime_process(6, 53)
  params = Params()
  channel = BigModelChannel(create=True)

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

  st = time.monotonic()
  cloudlog.warning("bigmodeld loading big model")
  try:
    model = ModelState(vipc_client_main.width, vipc_client_main.height, usbgpu=True)
  except Exception:
    cloudlog.exception("bigmodeld failed to load big model, exiting")
    return
  cloudlog.warning(f"bigmodeld loaded big model in {time.monotonic() - st:.1f}s")

  sm = SubMaster(["deviceState", "carState", "roadCameraState", "liveCalibration", "driverMonitoringState", "carControl", "liveDelay"])
  if demo:
    CP = get_demo_car_params()
  else:
    CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  long_delay = CP.longitudinalActuatorDelay + LONG_SMOOTH_SECONDS
  DH = DesireHelper()

  model_transform_main = np.zeros((3, 3), dtype=np.float32)
  model_transform_extra = np.zeros((3, 3), dtype=np.float32)
  buf_main, buf_extra = None, None
  meta_main = FrameMeta()
  meta_extra = FrameMeta()
  last_vipc_frame_id = 0
  run_count = 0

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
    lat_delay = sm["liveDelay"].lateralDelay + LAT_SMOOTH_SECONDS
    if sm.updated["liveCalibration"] and sm.seen['roadCameraState'] and sm.seen['deviceState']:
      device_from_calib_euler = np.array(sm["liveCalibration"].rpyCalib, dtype=np.float32)
      dc = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['roadCameraState'].sensor))]
      model_transform_main = get_warp_matrix(device_from_calib_euler, dc.ecam.intrinsics if main_wide_camera else dc.fcam.intrinsics, False).astype(np.float32)
      model_transform_extra = get_warp_matrix(device_from_calib_euler, dc.ecam.intrinsics, True).astype(np.float32)

    traffic_convention = np.zeros(2)
    traffic_convention[int(is_rhd)] = 1
    vec_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
    if desire >= 0 and desire < ModelConstants.DESIRE_LEN:
      vec_desire[desire] = 1

    vipc_dropped_frames = max(0, meta_main.frame_id - last_vipc_frame_id - 1)
    run_count += 1
    prepare_only = vipc_dropped_frames > 0

    bufs = {name: buf_extra if 'big' in name else buf_main for name in model.vision_input_names}
    transforms = {name: model_transform_extra if 'big' in name else model_transform_main for name in model.vision_input_names}
    frame_delay = DT_MDL
    action_delay = DT_MDL / 2
    lat_action_t = lat_delay + frame_delay + action_delay
    long_action_t = long_delay + frame_delay + action_delay
    inputs = {
      'desire_pulse': vec_desire,
      'traffic_convention': traffic_convention,
      'action_t': np.array([lat_action_t, long_action_t], dtype=np.float32),
    }

    model_output = model.run(bufs, transforms, inputs, prepare_only)
    if model_output is not None:
      channel.write(meta_main.frame_id, model_output)
      # drive our own desire from the model output so the big model gets the same lane change input
      desire_state = model_output['desire_state'][0].reshape(-1)
      lane_change_prob = desire_state[log.Desire.laneChangeLeft] + desire_state[log.Desire.laneChangeRight]
      DH.update(sm['carState'], sm['carControl'].latActive, lane_change_prob)
    last_vipc_frame_id = meta_main.frame_id


if __name__ == "__main__":
  try:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()
    main(demo=args.demo)
  except KeyboardInterrupt:
    cloudlog.warning("bigmodeld got SIGINT")
