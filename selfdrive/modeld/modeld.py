#!/usr/bin/env python3
import os
os.environ['GMMU'] = '0' # for usbgpu fast loading, noop for qcom
from tinygrad.tensor import Tensor
import time
import pickle
import numpy as np
import cereal.messaging as messaging
from cereal import log
from cereal.messaging import PubMaster
from msgq.visionipc import VisionBuf
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan, smooth_value, get_curvature_from_plan
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.selfdrive.modeld.compile_modeld import make_input_queues, WARP_INPUTS, POLICY_INPUTS
from openpilot.selfdrive.modeld.fill_model_msg import fill_model_msg, fill_driving_model_data, fill_pose_msg, PublishState
from openpilot.common.file_chunker import read_file_chunked, get_manifest_path
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
from openpilot.selfdrive.modeld.helpers import usbgpu_present, modeld_pkl_path, get_tg_input_devices
from openpilot.selfdrive.modeld.model_channel import ModelChannel, BIG_CHANNEL, SMALL_CHANNEL

PROCESS_NAME = "selfdrive.modeld.modeld"
SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')

LAT_SMOOTH_SECONDS = 0.0
LONG_SMOOTH_SECONDS = 0.3
MIN_LAT_CONTROL_SPEED = 0.3


def get_action_from_model(model_output: dict[str, np.ndarray], prev_action: log.ModelDataV2.Action,
                          lat_action_t: float, long_action_t: float, v_ego: float) -> log.ModelDataV2.Action:
  if 'action' not in model_output:
    plan = model_output['plan'][0]
    desired_accel, should_stop = get_accel_from_plan(plan[:,Plan.VELOCITY][:,0],
                                                     plan[:,Plan.ACCELERATION][:,0],
                                                     ModelConstants.T_IDXS,
                                                     action_t=long_action_t)
    desired_curvature = get_curvature_from_plan(plan[:,Plan.T_FROM_CURRENT_EULER][:,2],
                                                plan[:,Plan.ORIENTATION_RATE][:,2],
                                                ModelConstants.T_IDXS,
                                                v_ego,
                                                lat_action_t)
  else:
    desired_accel = model_output['action'][0,1]
    desired_curvature = model_output['action'][0,0] / (max(1.0, v_ego))**2
    should_stop = (v_ego < 0.3 and desired_accel < 0.1)
  desired_accel = smooth_value(desired_accel, prev_action.desiredAcceleration, LONG_SMOOTH_SECONDS)
  if v_ego > MIN_LAT_CONTROL_SPEED:
    desired_curvature = smooth_value(desired_curvature, prev_action.desiredCurvature, LAT_SMOOTH_SECONDS)
  else:
    desired_curvature = prev_action.desiredCurvature

  return log.ModelDataV2.Action(desiredCurvature=float(desired_curvature),
                                desiredAcceleration=float(desired_accel),
                                shouldStop=bool(should_stop))


class FrameMeta:
  frame_id: int = 0
  timestamp_sof: int = 0
  timestamp_eof: int = 0

  def __init__(self, vipc=None):
    if vipc is not None:
      self.frame_id, self.timestamp_sof, self.timestamp_eof = vipc.frame_id, vipc.timestamp_sof, vipc.timestamp_eof


class ModelState:
  prev_desire: np.ndarray  # for tracking the rising edge of the pulse

  def __init__(self, cam_w: int, cam_h: int, usbgpu: bool):
    input_devices = get_tg_input_devices(PROCESS_NAME, usbgpu)
    self.WARP_DEV, self.QUEUE_DEV = input_devices['WARP_DEV'], input_devices['QUEUE_DEV']
    jits = pickle.loads(read_file_chunked(modeld_pkl_path(usbgpu)))
    metadata = jits['metadata']
    self.input_shapes = metadata['input_shapes']
    self.vision_input_names = [k for k in self.input_shapes if 'img' in k]
    self.output_slices = metadata['output_slices']

    self.prev_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)

    self.frame_skip = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ
    self.input_queues, self.npy = make_input_queues(self.input_shapes, self.frame_skip, device=self.QUEUE_DEV)
    self.full_frames: dict[str, Tensor] = {}
    self._blob_cache: dict[int, Tensor] = {}
    self.parser = Parser()
    self.frame_buf_params = {k: get_nv12_info(cam_w, cam_h) for k in ('img', 'big_img')}
    self.run_policy = jits['run_policy']
    self.warp_enqueue = jits[(cam_w,cam_h)]

  def slice_outputs(self, model_outputs: np.ndarray, output_slices: dict[str, slice]) -> dict[str, np.ndarray]:
    parsed_model_outputs = {k: model_outputs[np.newaxis, v] for k,v in output_slices.items()}
    return parsed_model_outputs

  def run(self, bufs: dict[str, VisionBuf], transforms: dict[str, np.ndarray],
          inputs: dict[str, np.ndarray], prepare_only: bool) -> dict[str, np.ndarray] | None:
    for key in bufs.keys():
      ptr = np.frombuffer(bufs[key].data, dtype=np.uint8).ctypes.data
      yuv_size = self.frame_buf_params[key][3]
      # There is a ringbuffer of imgs, just cache tensors pointing to all of them
      cache_key = (key, ptr)
      if cache_key not in self._blob_cache:
        self._blob_cache[cache_key] = Tensor.from_blob(ptr, (yuv_size,), dtype='uint8', device=self.WARP_DEV)
      self.full_frames[key] = self._blob_cache[cache_key]

    # Model decides when action is completed, so desire input is just a pulse triggered on rising edge
    inputs['desire_pulse'][0] = 0
    self.npy['desire'][:] = np.where(inputs['desire_pulse'] - self.prev_desire > .99, inputs['desire_pulse'], 0)
    self.prev_desire[:] = inputs['desire_pulse']
    self.npy['traffic_convention'][:] = inputs['traffic_convention']
    self.npy['action_t'][:] = inputs['action_t']
    self.npy['tfm'][:,:] = transforms['img'][:,:]
    self.npy['big_tfm'][:,:] = transforms['big_img'][:,:]

    img, big_img = self.warp_enqueue(**{k: self.input_queues[k] for k in WARP_INPUTS}, frame=self.full_frames['img'], big_frame=self.full_frames['big_img'])

    if prepare_only:
      return None

    outs, = self.run_policy(
      **{k: self.input_queues[k] for k in POLICY_INPUTS if k in self.input_queues}, img=img, big_img=big_img
    )
    model_output = outs.numpy()[0]
    outputs_dict = self.parser.parse_outputs(self.slice_outputs(model_output, self.output_slices))
    self.npy['prev_feat'][:] = model_output[self.output_slices['hidden_state']]

    if SEND_RAW_PRED:
      outputs_dict['raw_pred'] = model_output.copy()
    return outputs_dict


BIG_MODEL_DEADLINE = 0.025  # wait for big only briefly past when small is ready, keeps the fallback frame on-time


def _publish(pm, publish_state, prev_action, payload):
  output = payload['output']
  action = get_action_from_model(output, prev_action, payload['lat_action_t'], payload['long_action_t'], payload['v_ego'])
  modelv2_send = messaging.new_message('modelV2')
  drivingdata_send = messaging.new_message('drivingModelData')
  posenet_send = messaging.new_message('cameraOdometry')
  fill_model_msg(modelv2_send, output, action, publish_state, payload['frame_id'], payload['frame_id_extra'],
                 payload['road_frame_id'], payload['frame_drop_ratio'], payload['timestamp_eof'],
                 payload['model_execution_time'], payload['live_calib_seen'])
  modelv2_send.modelV2.meta.laneChangeState = payload['lane_change_state']
  modelv2_send.modelV2.meta.laneChangeDirection = payload['lane_change_direction']
  modelv2_send.modelV2.usbGpu = payload['used_big']
  fill_driving_model_data(drivingdata_send, modelv2_send)
  fill_pose_msg(posenet_send, output, payload['frame_id'], payload['vipc_dropped_frames'],
                payload['timestamp_eof'], payload['live_calib_seen'])
  pm.send('modelV2', modelv2_send)
  pm.send('drivingModelData', drivingdata_send)
  pm.send('cameraOdometry', posenet_send)
  return action


def main(demo=False):
  # thin selector: bigmodeld and smallmodeld run in their own processes and write to a shared channel
  # modeld runs no model, publishes big when fresh else small, so losing the usbgpu just falls to small
  cloudlog.warning("modeld (selector) init")
  params = Params()
  _present = usbgpu_present()
  _compiled = os.path.isfile(get_manifest_path(modeld_pkl_path(usbgpu=True)))
  use_big = _present and _compiled
  params.put_bool("UsbGpuPresent", _present)
  params.put_bool("UsbGpuCompiled", _compiled)
  params.put_bool("UsbGpuActive", False)

  config_realtime_process([0, 1, 2, 3], 54)  # selector is light, core 7 is reserved for the big model

  pm = PubMaster(["modelV2", "drivingModelData", "cameraOdometry"])
  publish_state = PublishState()
  prev_action = log.ModelDataV2.Action()

  small_channel = None
  big_channel = None
  big_active = False
  last_published = -1
  # frameDropPerc must reflect our published cadence, not a worker's internal frame skipping
  # (which spikes during big's warmup and would falsely trip modeldLagging). small fills any gap
  frame_dropped_filter = FirstOrderFilter(0., 10., 1. / ModelConstants.MODEL_RUN_FREQ)
  run_count = 0
  big_used_count = 0
  big_done = False  # latch: big stalled catastrophically, do not use it again until next ignition
  big_miss = 0      # consecutive frames big failed to deliver while it was the active source
  cloudlog.warning(f"modeld selector starting (use_big={use_big} usbgpu_present={_present} compiled={_compiled})")

  while True:
    if small_channel is None:
      try:
        small_channel = ModelChannel(SMALL_CHANNEL, create=False)
      except OSError:
        time.sleep(0.05)
        continue
    if big_channel is None and use_big:
      try:
        big_channel = ModelChannel(BIG_CHANNEL, create=False)
      except OSError:
        big_channel = None

    # pace off the small model: it produces every frame and is the guaranteed output
    fid = small_channel.peek_frame_id()
    if fid is None or fid == last_published:
      time.sleep(0.0005)
      continue
    target = fid
    t_start = time.perf_counter()

    payload = None
    used_big = False
    big_peek = None  # last frame_id seen on the big channel, for the heartbeat
    # try big unless it has catastrophically failed this ignition. wait only while it is on this frame
    # or one behind. if it is further behind it has not delivered this frame, so fall through to small.
    if big_channel is not None and use_big and not big_done:
      deadline = t_start + BIG_MODEL_DEADLINE
      while time.perf_counter() < deadline:
        bfid = big_channel.peek_frame_id()
        big_peek = bfid
        if bfid == target:
          got = big_channel.read()
          if got is not None and got[0] == target:
            payload = got[1]
            used_big = True
          break
        if bfid is None or (bfid != target and bfid != target - 1):
          break
        time.sleep(0.0005)

    if payload is None:
      got = small_channel.read()
      if got is not None and got[0] == target:
        payload = got[1]

    # latched source: switch to big ONCE it first delivers a frame, then stay on big. fall back to
    # small ONLY after big stalls for a while (catastrophic), then never use big again this ignition.
    # any single big miss is already filled with small above, so frames are never dropped and the
    # source flag does not flap. next ignition starts fresh from small.
    if used_big:
      big_miss = 0
      if not big_active:
        big_active = True
        params.put_bool("UsbGpuActive", True)
        cloudlog.warning(f"modeld switched to BIG model at frame {target}")
    elif big_active:
      big_miss += 1
      if big_miss >= 10:  # ~0.5s of big not delivering = catastrophic
        big_active = False
        big_done = True
        params.put_bool("UsbGpuActive", False)
        cloudlog.warning(f"modeld big model stalled, staying on small until next ignition (frame {target})")

    if payload is not None:
      # drops from our own published cadence, overriding the worker's value. nonzero only when
      # small itself falls behind, the real "can't keep up" case
      selector_dropped = max(0, target - last_published - 1) if last_published >= 0 else 0
      frames_dropped = frame_dropped_filter.update(min(selector_dropped, 10))
      # ignore the first 40 frames (~2s) while both workers ramp to 20Hz
      if run_count < 40:
        frame_dropped_filter.x = 0.
        frames_dropped = 0.
      run_count += 1
      if used_big:
        big_used_count += 1
      # heartbeat in the rlog: how often big is actually used, and what we last saw on the big channel
      if run_count % 100 == 0:
        cloudlog.warning(f"modeld selector: big_used={big_used_count}/{run_count} last_big_peek={big_peek} target={target}")
      payload['vipc_dropped_frames'] = selector_dropped
      payload['frame_drop_ratio'] = frames_dropped / (1 + frames_dropped)
      payload['used_big'] = used_big
      prev_action = _publish(pm, publish_state, prev_action, payload)
      # only advance once we actually published, so a torn read retries this frame next poll
      # instead of skipping it
      last_published = target


if __name__ == "__main__":
  try:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='A boolean for demo mode.')
    args = parser.parse_args()
    main(demo=args.demo)
  except KeyboardInterrupt:
    cloudlog.warning("got SIGINT")
