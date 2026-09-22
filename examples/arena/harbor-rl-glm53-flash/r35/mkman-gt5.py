"""Build manifest-adebt-gt5.jsonl: chains of the 834 manifest with >5 segments.

le5 was built as the byte-exact subset of 834 with segcount<=5, keeping row
order and bytes. So gt5 is the exact set difference 834 - le5 (segcount>5).
No segcounts.json needed. Kept rows keep their 834 order and bytes.
"""
import hashlib, json

SRC = "/home/guparpit/glm53-prep/manifest-adebt-834.jsonl"
LE5 = "/home/guparpit/glm53-prep/manifest-adebt-le5.jsonl"
OUT = "/home/guparpit/glm53-prep/manifest-adebt-gt5.jsonl"

def tid(raw):
    return json.loads(raw)["lakefs_uri"].rstrip("/").split("/")[-1]

le5_ids = {tid(r) for r in open(LE5, "rb")}
kept = [r for r in open(SRC, "rb") if tid(r) not in le5_ids]
with open(OUT, "wb") as f:
    f.writelines(kept)
md5 = hashlib.md5(open(OUT, "rb").read()).hexdigest()
print(f"kept {len(kept)} chains -> {OUT} md5 {md5}")
assert len(kept) == 834 - 556, len(kept)
