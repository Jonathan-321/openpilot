#!/usr/bin/env python3
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.modeld.model_channel import SMALL_CHANNEL
from openpilot.selfdrive.modeld.model_worker import run


def main(demo=False):
  # background pool [0,1,2,3]: big now owns core 7 and camerad owns core 6. small is light (qcom, no
  # usb) so it keeps up here, and as its own process it always produces no matter what big does.
  try:
    run(usbgpu=False, channel_path=SMALL_CHANNEL, core=[0, 1, 2, 3], demo=demo)
  except Exception:
    cloudlog.exception("smallmodeld crashed")  # the launcher only sends crashes to sentry, log to rlog too
    raise


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--demo', action='store_true')
  args = parser.parse_args()
  try:
    main(demo=args.demo)
  except KeyboardInterrupt:
    cloudlog.warning("smallmodeld got SIGINT")
