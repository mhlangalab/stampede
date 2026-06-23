def test_import():
    import stampede as st

    assert hasattr(st.config, "keys")
