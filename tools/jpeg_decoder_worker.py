"""CPU Pillow decoder over binary stdio for independent cross-env audits.

No filesystem writes, network, torch, or model inference. The parent sends a
length-prefixed JPEG and receives canonical RGB bytes and dimensions.
"""
from __future__ import annotations

import io
import json
import platform
import struct
import sys

import PIL
from PIL import Image, features


def read_exact(stream, count):
    chunks = []
    while count:
        block = stream.read(count)
        if not block:
            raise EOFError('decoder input ended inside a request')
        chunks.append(block)
        count -= len(block)
    return b''.join(chunks)


def main():
    incoming, outgoing = sys.stdin.buffer, sys.stdout.buffer
    metadata = json.dumps({'python': platform.python_version(), 'pillow': PIL.__version__,
                           'libjpeg': features.version('jpg')}).encode('utf-8')
    outgoing.write(struct.pack('<I', len(metadata)) + metadata)
    outgoing.flush()
    while True:
        header = incoming.read(4)
        if not header:
            return
        if len(header) != 4:
            raise EOFError('decoder request header truncated')
        length = struct.unpack('<I', header)[0]
        if not 0 < length <= 20 * 1024 * 1024:
            raise ValueError('unexpected JPEG request length')
        payload = read_exact(incoming, length)
        with Image.open(io.BytesIO(payload)) as source:
            image = source.convert('RGB')
            if not 0 < image.width * image.height <= 16 * 1024 * 1024:
                raise ValueError('unexpected source image dimensions')
            rgb = image.tobytes()
            outgoing.write(struct.pack('<III', image.height, image.width, len(rgb)) + rgb)
            outgoing.flush()


if __name__ == '__main__':
    main()
