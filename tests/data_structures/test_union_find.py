from algolab.data_structures.union_find import UnionFind


def test_union_find_basic():
    uf = UnionFind(5)
    assert uf.union(0, 1)
    assert uf.union(1, 2)
    assert not uf.union(0, 2)
    assert uf.connected(0, 2)
    assert not uf.connected(0, 3)
    assert uf.count == 3
