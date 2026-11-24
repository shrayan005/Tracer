import pytest

from app.integrations.storage.file_storage import LocalFileStorage, make_key, sha256_of

PDF = b"%PDF-1.7\nfake pdf bytes\n%%EOF"


def test_make_key_layout():
    sha = sha256_of(PDF)
    assert make_key(sha, ".PDF") == f"{sha[:2]}/{sha[2:4]}/{sha}.pdf"


async def test_save_read_exists_delete(tmp_path):
    storage = LocalFileStorage(tmp_path)
    key = make_key(sha256_of(PDF), "pdf")
    assert not await storage.exists(key)
    await storage.save(key, PDF)
    assert await storage.exists(key)
    assert await storage.read(key) == PDF
    assert (tmp_path / key).is_file()
    assert not list(tmp_path.rglob("*.tmp"))          # no leftover temp files
    await storage.delete(key)
    assert not await storage.exists(key)
    await storage.delete(key)                          # deleting twice is fine


async def test_same_file_stored_once(tmp_path):
    storage = LocalFileStorage(tmp_path)
    key = make_key(sha256_of(PDF), "pdf")
    await storage.save(key, PDF)
    await storage.save(key, PDF)
    assert len(list(tmp_path.rglob("*.pdf"))) == 1


@pytest.mark.parametrize("bad_key", ["../../etc/passwd", "a/b/c.pdf", "/etc/passwd",
                                     "3f/a9/" + "0" * 64 + ".pdf/../../x", "3F/A9/" + "A" * 64 + ".pdf"])
async def test_rejects_keys_not_made_by_make_key(tmp_path, bad_key):
    storage = LocalFileStorage(tmp_path)
    with pytest.raises(ValueError):
        await storage.save(bad_key, PDF)
    with pytest.raises(ValueError):
        await storage.read(bad_key)
