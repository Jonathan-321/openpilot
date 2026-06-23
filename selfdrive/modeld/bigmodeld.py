#!/usr/bin/env python3
import os
import time
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.modeld.helpers import usbgpu_present, modeld_pkl_path
from openpilot.common.file_chunker import get_manifest_path
from openpilot.selfdrive.modeld.model_channel import BIG_CHANNEL
from openpilot.selfdrive.modeld.model_worker import run


def main(demo=False):
  present = usbgpu_present()
  compiled = os.path.isfile(get_manifest_path(modeld_pkl_path(usbgpu=True)))
  if not (present and compiled):
    # no big model to run. idle, do NOT return: a returning process trips the manager's
    # "bigmodeld not running" alert. present vs compiled tells us which one is missing.
    cloudlog.warning(f"bigmodeld idling, no big model (usbgpu_present={present} compiled={compiled})")
    while True:
      time.sleep(1)
  # core 7, shared with the small model. big loads and warms up at low priority so it never starves
  # small, then runs at higher priority than small (53 vs 52) once warm (see model_worker).
  try:
    run(usbgpu=True, channel_path=BIG_CHANNEL, core=7, priority=53, demo=demo)
  except Exception:
    # any crash before the worker's own model-load/run guards (eg vision/setup phase) must not exit:
    # a returning bigmodeld trips the manager "not running" alert which soft-disables even though
    # small is healthy. park until the next ignition, the selector stays on small.
    cloudlog.exception("bigmodeld crashed, parking until next ignition cycle")
    while True:
      time.sleep(1)


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--demo', action='store_true')
  args = parser.parse_args()
  try:
    main(demo=args.demo)
  except KeyboardInterrupt:
    cloudlog.warning("bigmodeld got SIGINT")
