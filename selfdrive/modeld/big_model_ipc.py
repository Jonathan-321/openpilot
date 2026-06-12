# single-writer/single-reader shared memory to pass the big model output from bigmodeld to modeld
import mmap
import os
import pickle
import struct

SHM_PATH = "/dev/shm/openpilot_bigmodel"
SHM_SIZE = 4 * 1024 * 1024
HEADER = struct.Struct("<QqQ")  # seq, frame_id, length


class BigModelChannel:
  def __init__(self, create: bool):
    if create:
      fd = os.open(SHM_PATH, os.O_CREAT | os.O_RDWR, 0o600)
      os.ftruncate(fd, SHM_SIZE)
    else:
      fd = os.open(SHM_PATH, os.O_RDWR)
    self.mm = mmap.mmap(fd, SHM_SIZE)
    os.close(fd)
    if create:
      self.mm[:HEADER.size] = HEADER.pack(0, -1, 0)

  def write(self, frame_id: int, model_output: dict) -> None:
    data = pickle.dumps(model_output, protocol=pickle.HIGHEST_PROTOCOL)
    assert HEADER.size + len(data) <= SHM_SIZE, "big model output too large for shm"
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
