#!/usr/bin/env python3
import os
import time

import pyray as rl
from cereal import messaging
from openpilot.system.hardware import TICI
from openpilot.common.realtime import config_realtime_process, set_core_affinity
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.layouts.main import MainLayout
from openpilot.selfdrive.ui.mici.layouts.main import MiciMainLayout
from openpilot.selfdrive.ui.ui_state import ui_state

BIG_UI = gui_app.big_ui()


def main():
  cores = {5, }
  config_realtime_process(0, 51)

  pm = messaging.PubMaster(['uiDebug'])

  gui_app.init_window("UI")
  if BIG_UI:
    MainLayout()
  else:
    MiciMainLayout()

  last_time = time.monotonic()

  for should_render, cpu_time in gui_app.render():
    now = time.monotonic()
    frame_time = now - last_time
    last_time = now

    uiDebug = messaging.new_message("uiDebug")
    uiDebug.uiDebug.cpuTimeMillis = cpu_time * 1000
    uiDebug.uiDebug.frameTimeMillis = frame_time * 1000
    uiDebug.uiDebug.frameTimeMillisRaylib = rl.get_frame_time() * 1000
    pm.send("uiDebug", uiDebug)

    ui_state.update()
    if should_render:
      # reaffine after power save offlines our core
      if TICI and os.sched_getaffinity(0) != cores:
        try:
          set_core_affinity(list(cores))
        except OSError:
          pass


if __name__ == "__main__":
  main()
