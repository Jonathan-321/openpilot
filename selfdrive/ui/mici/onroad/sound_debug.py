import pyray as rl
from openpilot.common.params import Params
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget
from openpilot.selfdrive.ui.ui_state import ui_state


class SoundDebug(Widget):
  def __init__(self):
    super().__init__()
    self._params = Params()
    self._enabled = self._params.get_bool("ShowSoundDebug")
    self._font = gui_app.font(FontWeight.MEDIUM)

  def _render(self, rect):
    if gui_app.frame % 30 == 0:
      self._enabled = self._params.get_bool("ShowSoundDebug")
    if not self._enabled:
      return

    sd = ui_state.sm["soundDebug"]
    alert = str(sd.alert).split(".")[-1]
    rows = [
      ("volume", f"{sd.volume * 100:.0f}%"),
      ("ambient", f"{sd.ambientDb:.1f} dB"),
      ("mic raw", f"{sd.rawDb:.1f} dB"),
      ("alert", alert),
    ]
    fs, pad, gap, lh = 32, 14, 24, 40

    label_w = max(measure_text_cached(self._font, lbl, fs).x for lbl, _ in rows)
    value_w = max(measure_text_cached(self._font, val, fs).x for _, val in rows)
    w = 2 * pad + label_w + gap + value_w
    h = 2 * pad + lh * len(rows)
    x0, y0 = rect.x + rect.width - w - 6, rect.y + 6

    rl.draw_rectangle_rounded(rl.Rectangle(x0, y0, w, h), 0.12, 10, rl.Color(0, 0, 0, 210))
    for i, (label, value) in enumerate(rows):
      y = y0 + pad + lh * i
      rl.draw_text_ex(self._font, label, rl.Vector2(x0 + pad, y), fs, 0, rl.Color(170, 170, 170, 255))
      vx = x0 + w - pad - measure_text_cached(self._font, value, fs).x
      rl.draw_text_ex(self._font, value, rl.Vector2(vx, y), fs, 0, rl.WHITE)
