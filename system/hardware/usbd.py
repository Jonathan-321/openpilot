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
HIST_LEN = 24          # 240ms of LTSSM in every publish

DEVICE = "/sys/bus/usb/devices/4-1"  # aux USB
PORT = "/sys/bus/usb/devices/usb4/4-0:1.0/usb4-port1"

# debugfs link_state strings
LTSSM = {
  "U0": "u0", "U1": "u1", "U2": "u2", "U3": "u3",
  "SS.Disabled": "ssDisabled", "Rx.Detect": "rxDetect", "SS.Inactive": "ssInactive",
  "Poll": "poll", "Recovery": "recovery", "HRESET": "hotReset",
  "Compliance": "compliance", "Loopback": "loopback", "Reset": "reset", "Resume": "resume",
  "USB device is powered off": "poweredOff",   # controller in LPM -> a sleep signal
}
# history ring
ORD = {n: i for i, n in enumerate([
  "unknown", "u0", "u1", "u2", "u3", "ssDisabled", "rxDetect", "ssInactive",
  "poll", "recovery", "hotReset", "compliance", "loopback", "reset", "resume", "poweredOff",
])}
# states whose ENTRY we count
ENTRY_COUNTER = {
  "u1": "u1EntryCount", "u2": "u2EntryCount", "u3": "u3EntryCount",
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


def over_current_count() -> int:
  return read_int(os.path.join(PORT, "over_current_count"))


def find_controller() -> tuple[str | None, str | None]:
  """Resolve the dwc3 controller behind DEVICE"""
  ctrl = None
  m = re.search(r"([0-9a-f]+\.dwc3)", os.path.realpath(DEVICE))
  if m:
    ctrl = m.group(1)
  link_state = f"/sys/kernel/debug/{ctrl}/link_state" if ctrl else None
  if not link_state or not os.path.exists(link_state):
    cands = glob.glob("/sys/kernel/debug/*dwc3*/link_state")
    link_state = cands[0] if cands else None
  ctrl_sysfs = f"/sys/bus/platform/devices/{ctrl}" if ctrl else None
  return link_state, ctrl_sysfs


def read_vbus_mv() -> int:
  for p in ("/sys/class/power_supply/usb/voltage_now", os.path.join(PORT, "../voltage_now")):
    uv = read_int(p, -1)
    if uv >= 0:
      return uv // 1000
  return 0


def main():
  pm = messaging.PubMaster(['usbState'])
  rk = Ratekeeper(POLL_HZ, print_delay_threshold=None)
  link_state_path, ctrl_sysfs = find_controller()
  cloudlog.event("usbd_start", link_state=link_state_path, ctrl=ctrl_sysfs, device=DEVICE)

  counts = dict.fromkeys(ENTRY_COUNTER.values(), 0)
  disconnect_count = 0
  hist: deque[int] = deque(maxlen=HIST_LEN)
  prev_ltssm = "unknown"
  last_non_u0 = "unknown"
  last_transition_ns = 0
  was_connected = False
  tick = 0

  while True:
    # fast LTSSM sampling to detect edges
    if link_state_path is not None:
      raw = read(os.path.basename(link_state_path), os.path.dirname(link_state_path))
      ltssm = LTSSM.get(raw, "unknown") if raw is not None else "unknown"
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

      # connect / disconnect
      if was_connected and not connected:
        disconnect_count += 1
        cloudlog.event("usb_disconnected", count=disconnect_count, lastNonU0=last_non_u0,
                       hist=list(hist), recovery=counts["recoveryCount"], rxDetect=counts["rxDetectCount"])
      elif connected and not was_connected:
        cloudlog.event("usb_connected", idVendor=read("idVendor"), idProduct=read("idProduct"), speed=speed)
      was_connected = connected

      msg = messaging.new_message('usbState', valid=True)
      s = msg.usbState
      s.connected = connected
      s.speedMbps = speed_mbps
      s.pmActive = read("power/runtime_status") == "active"
      s.disconnectCount = disconnect_count
      s.overCurrentCount = over_current_count()

      # LTSSM spine
      s.ltssmState = prev_ltssm
      s.lastNonU0State = last_non_u0
      s.lastTransitionMonoTime = last_transition_ns
      s.ltssmHistory = list(hist)

      # transition counters
      s.u1EntryCount = counts["u1EntryCount"]
      s.u2EntryCount = counts["u2EntryCount"]
      s.u3EntryCount = counts["u3EntryCount"]
      s.recoveryCount = counts["recoveryCount"]
      s.rxDetectCount = counts["rxDetectCount"]
      s.ssInactiveCount = counts["ssInactiveCount"]
      s.poweredOffCount = counts["poweredOffCount"]

      # SI counters
      if ctrl_sysfs is not None:
        s.linkErrorCount = read_int(os.path.join(ctrl_sysfs, "link_error_count"))
        s.ssResetCount = read_int(os.path.join(ctrl_sysfs, "ss_reset_count"))

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
