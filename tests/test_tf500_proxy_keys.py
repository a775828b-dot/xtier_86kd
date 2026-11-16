"""Key creation, quotas, expiry and deletion without live server requests."""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
import tf500_proxy_keys as keys


class KeyTests(unittest.TestCase):
    def test_create_quota_expiry_delete(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            master = root / "master"
            master.write_text("master-test\n", encoding="utf-8")
            with patch.object(keys, "MASTER_FILE", master), \
                 patch.object(keys, "KEYS_FILE", root / "keys.json"), \
                 patch.object(keys, "ACCESS_LOG", root / "access.jsonl"):
                item = keys.create("local client", 1, 1)
                self.assertEqual(keys.authenticate(item["key"])[1], 200)
                self.assertEqual(keys.authenticate("master-test")[1], 200)
                public = keys.public_entries()
                self.assertEqual(len(public), 2)
                self.assertNotIn(item["key"], json.dumps(public))
                (root / "access.jsonl").write_text(json.dumps({"time": int(time.time()),
                    "key_id": item["id"], "status": 200, "input": 10, "output": 5}) + "\n",
                    encoding="utf-8")
                self.assertEqual(keys.authenticate(item["key"])[1], 402)
                self.assertEqual(keys.usage()["by_key"][item["id"]]["input"], 10)
                with self.assertRaises(ValueError):
                    keys.delete("master")
                keys.delete(item["id"])
                self.assertEqual(keys.authenticate(item["key"])[1], 401)

                expired = keys.create("one day", 1, 0)
                stored = keys._read_keys()
                stored[0]["expires_at"] = int(time.time()) - 1
                keys._write_keys(stored)
                self.assertEqual(keys.authenticate(expired["key"])[1], 401)


if __name__ == "__main__":
    unittest.main()
