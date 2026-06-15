#!/usr/bin/env python3
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.modeld.model_channel import SMALL_CHANNEL
from openpilot.selfdrive.modeld.model_worker import run


def main(demo=False):
  # core 7, shared with the big model but at slightly lower priority (52 vs 53) so big stays fast.
  # small is light (qcom, no usb) and runs in big's eGPU-wait gaps, holding 20Hz.
  try:
    run(usbgpu=False, channel_path=SMALL_CHANNEL, core=7, priority=52, demo=demo)
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
