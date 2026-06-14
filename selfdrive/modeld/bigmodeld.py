#!/usr/bin/env python3
import os
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.modeld.helpers import usbgpu_present, modeld_pkl_path
from openpilot.common.file_chunker import get_manifest_path
from openpilot.selfdrive.modeld.model_channel import BIG_CHANNEL
from openpilot.selfdrive.modeld.model_worker import run


def main(demo=False):
  if not (usbgpu_present() and os.path.isfile(get_manifest_path(modeld_pkl_path(usbgpu=True)))):
    cloudlog.warning("no usbgpu / big model, bigmodeld exiting")
    return
  run(usbgpu=True, channel_path=BIG_CHANNEL, core=6, demo=demo)


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--demo', action='store_true')
  args = parser.parse_args()
  try:
    main(demo=args.demo)
  except KeyboardInterrupt:
    cloudlog.warning("bigmodeld got SIGINT")
