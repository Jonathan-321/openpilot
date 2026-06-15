#!/usr/bin/env python3
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.modeld.model_channel import SMALL_CHANNEL
from openpilot.selfdrive.modeld.model_worker import run


def main(demo=False):
  # core 7 with the selector, like master's modeld. not core 6: camerad is pinned there
  # (camerad/main.cc) and sharing it starved the camera and this fallback
  run(usbgpu=False, channel_path=SMALL_CHANNEL, core=7, demo=demo)


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--demo', action='store_true')
  args = parser.parse_args()
  try:
    main(demo=args.demo)
  except KeyboardInterrupt:
    cloudlog.warning("smallmodeld got SIGINT")
