"""The DSA sparse attention kernel of miles 4716a367a (the r17 image), for the kdatp-dsa T1 test.

The three modules are ``git show 4716a367a:miles_plugins/models/glm5/ops/<file>``, byte for byte
(blobs 260b38ed, e226bb08, 022a6006). The PR #3608 image deletes that directory. The modules use only
relative imports, so this package loads from the harness dir with no edit.
"""

from .sparse_mla import SparseMLA

__all__ = ["SparseMLA"]
