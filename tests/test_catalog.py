import numpy as np
import pandas as pd
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
from askap import Filepair, Image
from catalog import ReferenceCatalog, Catalog, get_RACS_path


class GetRACSPathTest(TestCase):

    def test_on_ada(self):
        racs_path = get_RACS_path('ada.physics.usyd.edu.au')
        path = '/import/ada1/jpri6587/data/RACS-25asec-Mosaiced_Gaussians_Final_GalCut_v2021_03_01.fits'
        self.assertEqual(racs_path, path)

    def test_on_nimbus(self):
        racs_path = get_RACS_path('vast-data')
        path = '/home/joshoewahp/RACS-25asec-Mosaiced_Gaussians_Final_GalCut_v2021_03_01.fits'
        self.assertEqual(racs_path, path)

class ReferenceCatalogTest(TestCase):

    @classmethod
    def setUpClass(self):
        racspath = Path('tests/data/RACS_catalogue_test.fits')
        refpath = Path('tests/data/EPOCH08/COMBINED/STOKESI_SELAVY/VAST_0012+00A.EPOCH08.I.selavy.components.txt')
        self.racscat = ReferenceCatalog(racspath)
        self.refcat = ReferenceCatalog(refpath)

    def test_filepath_as_str(self):
        racscat = ReferenceCatalog('tests/data/RACS_catalogue_test.fits')
        self.assertEqual(len(racscat.sources), 5595)

    @patch('catalog.os.uname', return_value=['', ''])
    def test_autofind_refcat_exits_on_unknown_system(self, _):
        self.assertRaises(SystemExit, ReferenceCatalog)

    def test_racs_refcat(self):
        self.assertEqual(self.racscat.name, 'racs')

    def test_racs_refcat_nsources(self):
        self.assertEqual(len(self.racscat.sources), 5595)

    def test_racs_refcat_columns(self):
        self.assertIn('N_Gaus', self.racscat.sources.columns)
        self.assertIn('ra', self.racscat.sources.columns)
        self.assertIn('dec', self.racscat.sources.columns)
        self.assertIn('flux_int', self.racscat.sources.columns)
        self.assertIn('flux_peak', self.racscat.sources.columns)
        self.assertIn('field_centre_dist', self.racscat.sources.columns)
        self.assertIn('rms_image', self.racscat.sources.columns)

    def test_other_refcat(self):
        self.assertEqual(self.refcat.name, 'ref')
        
    def test_other_refcat_nsources(self):
        self.assertEqual(len(self.refcat.sources), 7976)

    def test_other_refcat_columns(self):
        self.assertIn('N_Gaus', self.refcat.sources.columns)
        self.assertIn('ra', self.refcat.sources.columns)
        self.assertIn('dec', self.refcat.sources.columns)
        self.assertIn('flux_int', self.refcat.sources.columns)
        self.assertIn('flux_peak', self.refcat.sources.columns)
        self.assertIn('field_centre_dist', self.refcat.sources.columns)
        self.assertIn('rms_image', self.refcat.sources.columns)


class CatalogTest(TestCase):

    @classmethod
    def setUpClass(self):

        rootpath = Path('tests/data/EPOCH08/COMBINED/')
        imagepath = rootpath / 'STOKESI_IMAGES/VAST_0012+00A.EPOCH08.I.fits'
        selavypath = rootpath / 'STOKESI_SELAVY/VAST_0012+00A.EPOCH08.I.selavy.components.txt'
        files = Filepair(imagepath, selavypath)

        racspath = Path('tests/data/RACS_catalogue_test.fits')
        racscat = ReferenceCatalog(racspath)

        self.image = Image(files, refcat=racscat.sources)

        # Mock data array
        self.image.data = np.zeros((4000, 4000))
        self.image.size_x = 4000
        self.image.size_y = 4000
        self.image._set_field_positions()


    def test_askap_defaults(self):
        catalog = Catalog(self.image, survey_name='askap')
        self.assertEqual(len(catalog.sources), 3525)
        self.assertEqual(catalog.frequency, 887491000)

    def test_askap_isolim_kwarg(self):
        catalog = Catalog(self.image, survey_name='askap', isolationlim=45)
        self.assertEqual(len(catalog.sources), 5633)

    def test_askap_snrlim_kwarg(self):
        catalog = Catalog(self.image, survey_name='askap', snrlim=7)
        self.assertEqual(len(catalog.sources), 2912)

    def test_askap_frequency_kwarg(self):
        catalog = Catalog(self.image, survey_name='askap', frequency=1200e9)
        self.assertEqual(catalog.frequency, 1200e9)

    def test_askap_no_sources(self):
        # Simulate no sources via an excessive isolation requirement
        # TODO: make this independent of isolationlim kwarg test
        self.assertRaises(AssertionError,
                          Catalog,
                          self.image, survey_name='askap', isolationlim=1000)
        
    def test_d2neighbour(self):
        catalog = Catalog(self.image, survey_name='askap')

        # Mock 3 sources with well-defined separations
        catalog.sources = pd.DataFrame({'ra': [0, 364, 0], 'dec': [1, 0, 3]})
        catalog._get_dist_to_neighbour()

        self.assertTrue(np.allclose(catalog.sources.d2neighbour.values,
                                    np.array([7200, 14842.47, 7200])))

    def test_icrf(self):
        catalog = Catalog(self.image, 'icrf')
        self.assertEqual(len(catalog.sources), 2)

    def test_nvss(self):
        catalog = Catalog(self.image, 'nvss')
        self.assertEqual(len(catalog.sources), 311)

    def test_racs(self):
        catalog = Catalog(self.image, 'racs')
        self.assertEqual(len(catalog.sources), 17)

    def test_sumss_no_sources(self):
        self.assertRaises(AssertionError,
                          Catalog,
                          self.image, survey_name='sumss')
