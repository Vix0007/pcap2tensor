"""Tests for feature preset resolution."""

import pytest

from pcap2tensor import AEGIS_6D, BASIC_3D, EXTENDED_10D, FULL_13D, get_preset, list_presets


def test_all_presets_listed():
    presets = list_presets()
    assert "aegis-6d" in presets
    assert "basic-3d" in presets
    assert "extended-10d" in presets
    assert "full-13d" in presets


def test_dimensions_match_names():
    assert sum(f.dim for f in BASIC_3D()) == 3
    assert sum(f.dim for f in AEGIS_6D()) == 6
    assert sum(f.dim for f in EXTENDED_10D()) == 10
    assert sum(f.dim for f in FULL_13D()) == 13


def test_case_and_separator_insensitive():
    a = get_preset("aegis-6d")
    b = get_preset("AEGIS_6D")
    c = get_preset("Aegis-6d")
    assert len(a) == len(b) == len(c) == 6


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        get_preset("nonexistent-preset")


def test_presets_return_fresh_instances():
    """Each call must return fresh, independent feature instances."""
    a = AEGIS_6D()
    b = AEGIS_6D()
    assert a is not b
    assert a[0] is not b[0]
