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
  # background pool: small keeps master's core 7, so big the add-on runs here. the eGPU does the
  # compute, big only needs cpu to feed it. watch the HUD Hz to confirm it holds 20Hz on the pool.
  try:
    run(usbgpu=True, channel_path=BIG_CHANNEL, core=[0, 1, 2, 3], priority=53, demo=demo)
  except Exception:
    cloudlog.exception("bigmodeld crashed")  # the launcher only sends crashes to sentry, log to rlog too
    raise


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--demo', action='store_true')
  args = parser.parse_args()
  try:
    main(demo=args.demo)
  except KeyboardInterrupt:
    cloudlog.warning("bigmodeld got SIGINT")
