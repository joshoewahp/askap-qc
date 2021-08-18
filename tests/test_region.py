from region import LowBandRegion, MidBandRegion
from unittest import TestCase


class BandRegion(TestCase):

    def setUp(self) -> None:
        self.lowreg = LowBandRegion(regions=['3', '4'])
        self.midreg = MidBandRegion(regions=['3', '4'])
        return super().setUp()
    
    def test_low_band_field_counts(self) -> None:
        regions = ['1', '2', '3', '4', '5', '6']
        nfields = [40, 28, 20, 19, 5, 1]
        for region, number in zip(regions, nfields):
            reg = LowBandRegion(regions=region)
            self.assertEqual(len(reg), number)

    def test_mid_band_field_counts(self) -> None:
        regions = ['1', '2', '3', '4', '5', '6']
        nfields = [43, 0, 4, 34, 8, 1]
        for region, number in zip(regions, nfields):
            reg = MidBandRegion(regions=region)
            self.assertEqual(len(reg), number)

    def test_invalid_region(self) -> None:
        self.assertRaises(ValueError, LowBandRegion, regions=['1', '7'])
        self.assertRaises(ValueError, MidBandRegion, regions=['1', '7'])
            
    def test_repr(self) -> None:
        self.assertEqual(repr(LowBandRegion(['4', '3'])),
                         "LowBandRegion(['3', '4'])")
        self.assertEqual(repr(MidBandRegion(['4', '3'])),
                         "MidBandRegion(['3', '4'])")

    def test_eq(self) -> None:
        self.assertTrue(LowBandRegion([3]) == LowBandRegion(['3']))

    def test_getitem(self) -> None:
        reg = LowBandRegion(['5', '6'])
        self.assertEqual(reg['5'], ['1724-31', '1739-25', '1752-31', '1753-18', '1806-25',])
        self.assertEqual(reg['6'], ['0127-73'])

    def test_iter(self) -> None:
        reg = LowBandRegion(['5', '6'])
        fields = [r for r in reg]
        self.assertEqual(fields,
                         ['1724-31', '1739-25', '1752-31', '1753-18', '1806-25', '0127-73'])
        

class TestMidBandRegion(TestCase):

    def test_mid_band(self) -> None:
        regions = ['1', '2', '3', '4', '5', '6']
        nfields = [43, 0, 4, 34, 8, 1]
        for region, number in zip(regions, nfields):
            reg = MidBandRegion(regions=region)
            self.assertEqual(len(reg), number)

    def test_invalid_region(self) -> None:
        self.assertRaises(ValueError, MidBandRegion, regions=['1', '7'])

    def test_repr(self) -> None:
        self.assertEqual(repr(MidBandRegion(['4', '3'])),
                         "MidBandRegion(['3', '4'])")
