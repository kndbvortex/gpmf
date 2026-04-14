"""Tests for core data structures (GradualItem, GradualPattern)."""
import pytest
from gpmf.core.data_structures import GradualItem, GradualPattern


class TestGradualItem:
    def test_valid_plus(self):
        gi = GradualItem(0, '+')
        assert gi.attribute_col == 0
        assert gi.symbol == '+'

    def test_valid_minus(self):
        gi = GradualItem(2, '-')
        assert gi.attribute_col == 2
        assert gi.symbol == '-'

    def test_invalid_symbol(self):
        with pytest.raises(ValueError):
            GradualItem(0, '*')

    def test_invert(self):
        gi = GradualItem(1, '+')
        inv = gi.inv_gi()
        assert inv.symbol == '-'
        assert inv.attribute_col == 1

    def test_string_representation(self):
        gi = GradualItem(3, '-')
        assert gi.to_string() == '3-'
        assert str(gi) == '3-'

    def test_equality(self):
        assert GradualItem(0, '+') == GradualItem(0, '+')
        assert GradualItem(0, '+') != GradualItem(0, '-')
        assert GradualItem(0, '+') != GradualItem(1, '+')

    def test_hashable(self):
        s = {GradualItem(0, '+'), GradualItem(0, '+'), GradualItem(1, '-')}
        assert len(s) == 2


class TestGradualPattern:
    def _make_pattern(self, items, support=0.5):
        gp = GradualPattern()
        for col, sym in items:
            gp.add_gradual_item(GradualItem(col, sym))
        gp.set_support(support)
        return gp

    def test_empty_pattern(self):
        gp = GradualPattern()
        assert len(gp) == 0
        assert gp.support == 0.0

    def test_add_items(self):
        gp = self._make_pattern([(0, '+'), (1, '-')])
        assert len(gp) == 2

    def test_support_rounded(self):
        gp = GradualPattern()
        gp.set_support(0.123456789)
        assert gp.support == 0.123

    def test_to_string(self):
        gp = self._make_pattern([(0, '+'), (2, '-')])
        assert '0+' in gp.to_string()
        assert '2-' in gp.to_string()

    def test_to_dict(self):
        gp = self._make_pattern([(0, '+')], support=0.75)
        d = gp.to_dict()
        assert d['support'] == 0.75
        assert '0+' in d['pattern']

    def test_add_items_from_list(self):
        gp = GradualPattern()
        gp.add_items_from_list(['0+', '1-', '2+'])
        assert len(gp) == 3
        assert gp.gradual_items[1].symbol == '-'
