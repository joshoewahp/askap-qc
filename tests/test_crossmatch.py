import os
from unittest import TestCase
from pathlib import Path

from askap import Image, Filepair
from catalog import Catalog, ReferenceCatalog
from crossmatch import Crossmatch


class CrossmatchTest(TestCase):

    @classmethod
    def setUpClass(self):
        rootpath = Path('data/testdata/EPOCH08/COMBINED/')
        imagepath = rootpath / 'STOKESI_IMAGES/VAST_0012+00A.EPOCH08.I.fits'
        selavypath = rootpath / 'STOKESI_SELAVY/VAST_0012+00A.EPOCH08.I.selavy.components.txt'
        files = Filepair(imagepath, selavypath)

        racspath = Path('data/testdata/RACS_catalogue_test.fits')
        racscat = ReferenceCatalog(racspath)

        self.image = Image(files, refcat=racscat.sources)
        self.askap_cat = Catalog(self.image, survey_name='askap')
        self.nvss_cat = Catalog(self.image, survey_name='nvss')
        self.icrf_cat = Catalog(self.image, survey_name='icrf')

    def setUp(self):
        self.cm = Crossmatch(self.askap_cat, self.nvss_cat)

    def test_repr(self):
        self.assertEqual(repr(self.cm), "<Crossmatch: askap-nvss>" )

    def test_base_comp_name_order(self):
        self.assertEqual(self.cm.base_cat.survey_name, 'askap')
        self.assertEqual(self.cm.comp_cat.survey_name, 'nvss')

    def test_base_comp_name_order(self):
        cm = Crossmatch(self.askap_cat, self.icrf_cat)
        self.assertEqual(cm.comp_cat.survey_name, 'icrf')
        self.assertTrue(cm.matches.flux_int_ratio.isna().all())
        self.assertTrue(cm.matches.flux_peak_ratio.isna().all())


    def test_number_matches(self):
        self.assertEqual(len(self.cm.matches), 1268)

    def test_no_matches_raises_error(self):
        self.assertRaises(AssertionError,
                          Crossmatch,
                          self.askap_cat, self.nvss_cat, maxoffset=0)

    def test_save_matchdf(self):
        os.makedirs('testsavedir', exist_ok=True)
        self.cm.save('testsavedir')
        self.assertTrue(os.path.exists('testsavedir/0012+00A_nvss-askap.csv'))
        os.system('rm -r testsavedir')

