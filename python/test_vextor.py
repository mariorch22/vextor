"""Smoke test for vextor Python bindings."""

import os
import shutil
import tempfile

import numpy as np


def test_insert_and_search():
    import vextor

    db = vextor.Database(dimensions=4, segment_capacity=100)

    db.insert(user_id=1, vector=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    db.insert(user_id=2, vector=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    db.insert(user_id=3, vector=np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))

    assert db.size == 3
    assert db.dimensions == 4

    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = db.search(query=query, k=1)

    assert len(results) == 1
    assert results[0][0] == 1  # user_id
    assert results[0][1] == 0.0  # distance
    print("insert_and_search: PASSED")


def test_insert_batch():
    import vextor

    db = vextor.Database(dimensions=3, segment_capacity=100)

    ids = np.array([10, 20, 30], dtype=np.uint64)
    vecs = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    db.insert_batch(user_ids=ids, vectors=vecs)
    assert db.size == 3

    query = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    results = db.search(query=query, k=1)
    assert results[0][0] == 20
    print("insert_batch: PASSED")


def test_save_and_load():
    import vextor

    tmpdir = tempfile.mkdtemp(prefix="vextor_test_")
    db_path = os.path.join(tmpdir, "testdb")

    try:
        db = vextor.Database(dimensions=4, segment_capacity=100, path=db_path)
        db.insert(user_id=42, vector=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        db.insert(user_id=43, vector=np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32))
        db.save()

        db2 = vextor.Database.load(db_path)
        assert db2.size == 2

        query = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        results = db2.search(query=query, k=1)
        assert results[0][0] == 42
        print("save_and_load: PASSED")
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    test_insert_and_search()
    test_insert_batch()
    test_save_and_load()
    print("\nAll Python tests passed!")