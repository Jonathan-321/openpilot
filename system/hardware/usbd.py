#!/usr/bin/env python3
import os
import re
import glob
import time
from collections import deque

from cereal import messaging
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog

POLL_HZ = 100          # LTSSM sample rate
PUB_HZ = 10            # cereal publish rate
PUB_DIV = POLL_HZ // PUB_HZ
HIST_LEN = 24          # ~240ms of LTSSM per publish

DEVICE = "/sys/bus/usb/devices/4-1"  # aux USB (eGPU)
PORT = "/sys/bus/usb/devices/usb4/4-0:1.0/usb4-port1"

# xHCI PORTSC Port Link State (bits 8:5) -> cereal LtssmState
PLS = ["u0", "u1", "u2", "u3", "ssDisabled", "rxDetect", "ssInactive", "poll",
       "recovery", "hotReset", "compliance", None, None, None, None, "resume"]
# history ring ordinals (cereal LtssmState)
ORD = {n: i for i, n in enumerate([
  "unknown", "u0", "u1", "u2", "u3", "ssDisabled", "rxDetect", "ssInactive",
  "poll", "recovery", "hotReset", "compliance", "loopback", "reset", "resume", "poweredOff"])}
# states whose ENTRY we count (best-effort: sampled, brief events may be missed)
ENTRY_COUNTER = {
  "recovery": "recoveryCount", "rxDetect": "rxDetectCount",
  "ssInactive": "ssInactiveCount", "poweredOff": "poweredOffCount",
}


def read(attr: str, base: str = DEVICE) -> str | None:
  try:
    with open(os.path.join(base, attr)) as f:
      return f.read().strip()
  except OSError:
    return None


def read_int(path: str, default: int = 0) -> int:
  try:
    with open(path) as f:
      return int(f.read().strip())
  except (OSError, ValueError):
    return default


def read_reg(attr: str, base: str) -> int | None:
  raw = read(attr, base)
  try:
    return int(raw, 16) if raw else None
  except ValueError:
    return None


def over_current_count() -> int:
  return read_int(os.path.join(PORT, "over_current_count"))


def find_controller() -> str | None:
  """Resolve the dwc3-msm (.ssusb) node exposing the host SS-port registers"""
  m = re.search(r"([0-9a-f]+\.ssusb)", os.path.realpath(DEVICE))
  if m:
    cand = f"/sys/bus/platform/devices/{m.group(1)}"
    if os.path.exists(os.path.join(cand, "portsc")):
      return cand
  for c in glob.glob("/sys/bus/platform/devices/*.ssusb"):
    if os.path.exists(os.path.join(c, "portsc")):
      return c
  return None


def read_vbus_mv() -> int:
  for p in ("/sys/class/power_supply/usb/voltage_now", os.path.join(PORT, "../voltage_now")):
    uv = read_int(p, -1)
    if uv >= 0:
      return uv // 1000
  return 0


def main():
  pm = messaging.PubMaster(['usbState'])
  rk = Ratekeeper(POLL_HZ, print_delay_threshold=None)
  ctrl = find_controller()
  cloudlog.event("usbd_start", ctrl=ctrl, device=DEVICE)

  counts = dict.fromkeys(ENTRY_COUNTER.values(), 0)
  disconnect_count = 0
  link_error_count = 0
  hist: deque[int] = deque(maxlen=HIST_LEN)
  prev_ltssm = "unknown"
  last_non_u0 = "unknown"
  last_transition_ns = 0
  was_connected = False
  tick = 0

  while True:
    # registers are only safe to read while the controller is powered/active
    pm_active = read("power/runtime_status") == "active"

    if ctrl is not None:
      if pm_active:
        reg = read_reg("portsc", ctrl)
        ltssm = (PLS[(reg >> 5) & 0xf] or "unknown") if reg else "unknown"
      else:
        ltssm = "poweredOff"
      hist.append(ORD[ltssm])
      if ltssm != prev_ltssm:
        last_transition_ns = time.monotonic_ns()
        if ltssm in ENTRY_COUNTER:            # count entries into states
          counts[ENTRY_COUNTER[ltssm]] += 1
        if ltssm != "u0":
          last_non_u0 = ltssm                 # latch the precursor
        prev_ltssm = ltssm

    if tick % PUB_DIV == 0:
      speed = read("speed")
      connected = speed is not None
      speed_mbps = int(speed) if (speed and speed.isdigit()) else 0

      # connect / disconnect (USB device presence)
      if was_connected and not connected:
        disconnect_count += 1
        cloudlog.event("usb_disconnected", count=disconnect_count, lastNonU0=last_non_u0)
      elif connected and not was_connected:
        cloudlog.event("usb_connected", idVendor=read("idVendor"), idProduct=read("idProduct"), speed=speed)
      was_connected = connected

      if ctrl is not None and pm_active:
        link_error_count = (read_reg("portli", ctrl) or 0) & 0xffff

      msg = messaging.new_message('usbState', valid=True)
      s = msg.usbState
      s.connected = connected
      s.speedMbps = speed_mbps
      s.pmActive = pm_active
      s.disconnectCount = disconnect_count
      s.overCurrentCount = over_current_count()

      # LTSSM spine
      s.ltssmState = prev_ltssm
      s.lastNonU0State = last_non_u0
      s.lastTransitionMonoTime = last_transition_ns
      s.ltssmHistory = list(hist)

      # link-event counters (best-effort) + hardware link error count
      s.recoveryCount = counts["recoveryCount"]
      s.rxDetectCount = counts["rxDetectCount"]
      s.ssInactiveCount = counts["ssInactiveCount"]
      s.poweredOffCount = counts["poweredOffCount"]
      s.linkErrorCount = link_error_count

      # sleep / power states
      s.runtimeSuspendedMs = read_int(os.path.join(DEVICE, "power/runtime_suspended_time"))
      s.lpmU1Enabled = read("power/usb3_hardware_lpm_u1") == "enabled"
      s.lpmU2Enabled = read("power/usb3_hardware_lpm_u2") == "enabled"

      # VBUS brownout discriminator
      s.vbusMv = read_vbus_mv()

      pm.send('usbState', msg)

    tick += 1
    rk.keep_time()


if __name__ == "__main__":
  main()
