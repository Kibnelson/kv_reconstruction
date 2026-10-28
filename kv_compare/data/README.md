# Vendored data

`higgs_grids.npz`: the HIGGS / EDEN vector-quantization grids used by AQUA-KV, copied unchanged (as float32 arrays)
from `aquakv/grids/` of https://github.com/goodevening13/aquakv at commit
e744183f2f3a4a90381f20a6975456871fd1ae29 (Apache License 2.0). Keys and the SHA-256 of the source files:

| Key | Source file | Shape | SHA-256 |
| --- | --- | --- | --- |
| `EDEN2_16` | `EDEN2-16.pt` | 16 x 2 (2 bits per value) | bdd4037ced580d137c780ad338ebdfdfd05d80b6ad5fc897ef2dc566a3c57697 |
| `EDEN2_64` | `EDEN2-64.pt` | 64 x 2 (3 bits per value) | 5838ad82ece96a56d886deccf6d8381debec0e79e146811c3b905067c003b7e3 |
| `EDEN2_256` | `EDEN2-256.pt` | 256 x 2 (4 bits per value) | 7f6fae561c7634ad63281038a23956043271388f383f8fa3425aacdd49f00fdc |
