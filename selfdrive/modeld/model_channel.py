# single-writer/single-reader shared memory to pass a model output from a model process to modeld
import mmap
import os
import pickle
import struct
from openpilot.common.swaglog import cloudlog

BIG_CHANNEL = "/dev/shm/openpilot_bigmodel"
SMALL_CHANNEL = "/dev/shm/openpilot_smallmodel"
SHM_SIZE = 8 * 1024 * 1024
HEADER = struct.Struct("<QqQ")  # seq, frame_id, length


class ModelChannel:
  def __init__(self, path: str, create: bool):
    if create:
      fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
      os.ftruncate(fd, SHM_SIZE)
    else:
      fd = os.open(path, os.O_RDWR)
    self.mm = mmap.mmap(fd, SHM_SIZE)
    os.close(fd)
    if create:
      self.mm[:HEADER.size] = HEADER.pack(0, -1, 0)

  def write(self, frame_id: int, model_output: dict) -> None:
    data = pickle.dumps(model_output, protocol=pickle.HIGHEST_PROTOCOL)
    if HEADER.size + len(data) > SHM_SIZE:
      # too large for the buffer. drop this frame instead of crashing the worker, the reader keeps
      # the previous frame and the next write recovers
      cloudlog.error(f"model output {len(data)} bytes exceeds shm {SHM_SIZE}, dropping frame {frame_id}")
      return
    seq = HEADER.unpack(self.mm[:HEADER.size])[0]
    # odd seq while writing, even when done, so the reader can detect a torn read
    HEADER.pack_into(self.mm, 0, seq + 1, frame_id, len(data))
    self.mm[HEADER.size:HEADER.size + len(data)] = data
    HEADER.pack_into(self.mm, 0, seq + 2, frame_id, len(data))

  def peek_frame_id(self) -> int | None:
    seq, frame_id, length = HEADER.unpack(self.mm[:HEADER.size])
    if seq == 0 or seq % 2 != 0 or length == 0:
      return None
    return frame_id

  def read(self) -> tuple[int, dict] | None:
    seq1, frame_id, length = HEADER.unpack(self.mm[:HEADER.size])
    if seq1 == 0 or seq1 % 2 != 0 or length == 0:
      return None
    data = bytes(self.mm[HEADER.size:HEADER.size + length])
    seq2 = HEADER.unpack(self.mm[:HEADER.size])[0]
    if seq1 != seq2:  # writer touched it mid-read, retry next poll
      return None
    try:
      return frame_id, pickle.loads(data)
    except Exception:
      return None
