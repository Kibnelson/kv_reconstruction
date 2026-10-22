"""Immutable source identities for the matched GPU baseline experiment."""
from pathlib import Path
import hashlib

AQUA_COMMIT = 'e744183f2f3a4a90381f20a6975456871fd1ae29'
EXANS_COMMIT = 'de6c38bcc5438475bd1abf1e9b3d154cef414ef6'
QUANTO_VERSION = '0.2.6'
AQUA_FILES = {
    'aquakv/quantizers.py': '3023fc2d489547b18dc239fb9cb8fa25d4d8468bd656e2c3103d92195ecbeda7',
    'aquakv/cache_utils.py': '6f1cab72358a046b7d2c802f265595bbca80205638968224c8ce5b1c955e116e',
    'aquakv/linear_utils.py': '5eed4b022e3fe0a7814ea2d5bb8f634dafe675d43c35e2e81cbfbad264d303ff',
    'LICENSE': 'e64e4a71b581b5c544162fe711eb176861d08b7c8922f54706890367d5303534'}
EXANS_CUDA = 'crates/openlake_kv_client/cuda'
EXANS_FILES = {
    EXANS_CUDA+'/src/expans_codec.cu': '4e43cd11f0ed294b467e90b96597988a9f26a84942e44de9031d769357d24dff',
    EXANS_CUDA+'/src/expans_kernels.cuh': '7380e32e8cdd5e9958ac58b68085aef0de8a4f08c21b2dc2d529e28aa86f18d0',
    EXANS_CUDA+'/include/openlake/expans_codec.h': 'a6f1b4cf993d45ddbf4d56a12c54af30080d347860dcd1be83e5c1189d98dba5',
    'LICENSE': '2cc5b0edcac471328efc7aad74b07002f4bafb4837bebd4456e9332b797b500b'}


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(root, files):
    for name, expected in files.items():
        if sha_file(Path(root)/name) != expected:
            raise ValueError('upstream baseline source hash mismatch: '+name)
