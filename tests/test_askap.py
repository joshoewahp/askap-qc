import copy
import numpy as np
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from askap import Filepair, Epoch, Region, Image
from catalog import ReferenceCatalog
from region import LowBandRegion, MidBandRegion


class EpochTest(TestCase):

    def setUp(self):
        self.epoch8path = Path('tests/data/EPOCH08/')
        self.region = Region([1])
        return super().setUp()

    def test_bad_tiletype_arg(self):
        self.assertRaises(ValueError,
                          Epoch,
                          self.epoch8path, tiletype='TILE', stokes='V', region=self.region, band='low')

    def test_combined_low_I_no_region(self):
        epoch = Epoch(self.epoch8path, tiletype='COMBINED', stokes='I', region=None, band='low')
        self.assertEqual(epoch.num_files, 2) 

    def test_combined_low_I_region(self):
        epoch = Epoch(self.epoch8path, tiletype='COMBINED', stokes='I', region=self.region, band='low')
        self.assertEqual(epoch.num_files, 1) 
        
    def test_tiles_low_I_no_region(self):
        epoch = Epoch(self.epoch8path, tiletype='TILES', stokes='I', region=None, band='low')
        self.assertEqual(epoch.num_files, 2) 

    def test_tiles_low_I_region(self):
        epoch = Epoch(self.epoch8path, tiletype='TILES', stokes='I', region=self.region, band='low')
        self.assertEqual(epoch.num_files, 1) 

    def test_combined_low_V_no_region(self):
        epoch = Epoch(self.epoch8path, tiletype='COMBINED', stokes='V', region=None, band='low')
        self.assertEqual(epoch.num_files, 2) 

    def test_combined_low_V_region(self):
        epoch = Epoch(self.epoch8path, tiletype='COMBINED', stokes='V', region=self.region, band='low')
        self.assertEqual(epoch.num_files, 1) 

    def test_tiles_low_V_no_region(self):
        self.assertRaises(NotImplementedError,
                          Epoch,
                          self.epoch8path, tiletype='TILES', stokes='V', region=None, band='low')

    def test_tiles_low_V_region(self):
        self.assertRaises(NotImplementedError,
                          Epoch,
                          self.epoch8path, tiletype='TILES', stokes='V', region=self.region, band='low')


class RegionTest(TestCase):

    def test_low_band(self):
        reg = Region(regions=['1'], band='low')
        self.assertIsInstance(reg, LowBandRegion)
            
    def test_mid_band(self):
        reg = Region(regions=['1'], band='mid')
        self.assertIsInstance(reg, MidBandRegion)

    def test_high_band(self):
        self.assertRaises(NotImplementedError, Region, regions=['1'], band='high')

    def test_bad_band_argument(self):
        self.assertRaises(ValueError, Region, regions='1', band='hi')


class ImageTest(TestCase):

    @classmethod
    def setUpClass(self):
        rootpath = Path('tests/data/EPOCH08/COMBINED/')

        imagepath_south = rootpath / 'STOKESI_IMAGES/VAST_0127-73A.EPOCH08.I.fits'
        selavypath_south = rootpath / 'STOKESI_SELAVY/VAST_0127-73A.EPOCH08.I.selavy.components.txt'
        imagepath_north = rootpath / 'STOKESI_IMAGES/VAST_0012+00A.EPOCH08.I.fits'
        selavypath_north = rootpath / 'STOKESI_SELAVY/VAST_0012+00A.EPOCH08.I.selavy.components.txt'

        self.refcat = ReferenceCatalog('tests/data/RACS_catalogue_test.fits')
        self.southfiles = Filepair(image=imagepath_south, selavy=selavypath_south)
        self.northfiles = Filepair(image=imagepath_north, selavy=selavypath_north)

        return super().setUpClass()

    def setUp(self):
        """
        Mock image data of 4000x4000 pixels and set up field positions
        to avoid including actual image data in tests directory.
        """

        self.northimage = Image(self.northfiles, refcat=self.refcat.sources)
        self.southimage = Image(self.southfiles, refcat=self.refcat.sources)

        self.northimage.data = np.zeros((4000, 4000))
        self.northimage.size_x = 4000
        self.northimage.size_y = 4000
        self.northimage._set_field_positions()

        self.southimage.data = np.zeros((4000, 4000))
        self.southimage.size_x = 4000
        self.southimage.size_y = 4000
        self.southimage._set_field_positions()

        return super().setUp()

    def check_column_names(self, catalogue):
        common_columns = {
            'ra', 'ra_err', 'dec', 'dec_err', 'maj_axis', 'min_axis', 'pos_ang',
            'flux_peak', 'flux_peak_err', 'flux_int', 'flux_int_err', 'rms_image'           
        }
        
        oneoff_columns = {
            '_r', 'N_Gaus', 'island_id', 'component_id', 'component_name', 'field_centre_dist'
        }

        required_columnset = oneoff_columns | common_columns
        catalogue_columnset = oneoff_columns | set(catalogue.columns)

        self.assertSetEqual(catalogue_columnset, required_columnset)
    
    def check_header_vals(self, image: Image):
        """Check value of header keywords are set correctly."""

        self.assertEqual(image.sbid, None)
        self.assertEqual(image.bmaj, 0.00364940923141675)
        self.assertEqual(image.bmin, 0.00356381661788453)
        self.assertEqual(image.bpa, 5.28573604455489)
        self.assertEqual(image.bcr_ra, 3.102094583333)
        self.assertEqual(image.bcr_dec, 0.003423722222222)
        self.assertEqual(image.frequency, 887491000)
    
    def test_repr(self):
        self.assertEqual(repr(self.northimage),
                         f"<Image: 'VAST_0012+00A.EPOCH08.I.fits'>")

    def test_init(self):
        self.assertEqual(self.northimage.filepair, self.northfiles)
        self.assertEqual(self.northimage.imagepath, self.northfiles.image)
        self.assertEqual(self.northimage.selavypath, self.northfiles.selavy)
        self.assertEqual(self.northimage.refcat.shape, self.refcat.sources.shape)

    def test_load(self):
        self.check_header_vals(self.northimage)
        self.assertEqual(self.northimage.data.shape, (4000, 4000))

    def test_load_no_data(self):
        image = Image(self.northfiles, refcat=self.refcat.sources, load_data=False)
        self.check_header_vals(image)
        self.assertEqual(image.data, None)
        
    def test_find_nearest_edge_axes_size_mismatched(self):

        # Copy northimage with mocked data array to avoid affecting other tests
        image = copy.copy(self.northimage)

        # Mock swapped image dimensions
        image.data = self.northimage.data[:, :-1]
        image.size_y = self.northimage.size_y - 1
        edge_pixels = image._find_nearest_edge()

        self.assertTrue((edge_pixels == [[2000, 3999]]).all())
    
    def test_parse_name(self):
        self.assertEqual(self.northimage.fieldname, '0012+00A')
        self.assertEqual(self.northimage.epoch, 'EPOCH08')
        self.assertEqual(self.northimage.stokes, 'I')

    def test_parse_nonmatching_name(self):
        rootpath = Path('tests/data/EPOCH05x/COMBINED/')
        imagepath = rootpath / 'STOKESI_IMAGES/VAST_0012+00A.EPOCH05x.I.conv.fits'
        selavypath = rootpath / 'STOKESI_SELAVY/VAST_0012+00A.EPOCH05x.I.selavy.components.txt'
        files = Filepair(image=imagepath, selavy=selavypath)

        image = Image(files, refcat=self.refcat.sources)

        self.assertEqual(image.fieldname, '0012+00A')
        self.assertEqual(image.epoch, None)
        self.assertEqual(image.stokes, None)
        
    def test_get_frequency_with_restfreq(self):
        freq = self.northimage._get_frequency()
        self.assertEqual(freq, 887.491e6)

    def test_get_frequency_with_crval3(self):
        restfreq = self.northimage.header.pop('RESTFREQ')
        self.northimage.header['CRVAL3'] = restfreq
        self.northimage.header['CTYPE3'] = 'FREQ'

        freq = self.northimage._get_frequency()
        self.assertEqual(freq, 887.491e6)

    def test_get_frequency_without_restfreq_or_crval3(self):
        self.northimage.header.pop('RESTFREQ')
        self.assertRaises(KeyError, self.northimage._get_frequency)

    def test_get_catalogue_askap(self):
        askap = self.northimage.get_catalogue('ASKAP')

        self.assertEqual(len(askap), 7976)
        self.check_column_names(askap)
        
    def test_get_catalogue_sumss_south(self):
        sumss = self.southimage.get_catalogue('SUMSS')

        self.assertEqual(len(sumss), 231)
        self.check_column_names(sumss)

    def test_get_catalogue_sumss_north(self):
        self.assertRaises(AssertionError, self.northimage.get_catalogue, 'SUMSS')

    def test_get_catalogue_nvss_south(self):
        self.assertRaises(AssertionError, self.southimage.get_catalogue, 'NVSS')

    def test_get_catalogue_nvss_north(self):
        nvss = self.northimage.get_catalogue('NVSS')

        self.assertEqual(len(nvss), 433)
        self.check_column_names(nvss)
        
    def test_get_catalogue_icrf(self):
        icrf = self.northimage.get_catalogue('ICRF')

        self.assertEqual(len(icrf), 2)
        self.check_column_names(icrf)

    def test_get_catalogue_racs(self):
        racs = self.northimage.get_catalogue('RACS')

        self.assertEqual(len(racs), 50)
        self.check_column_names(racs)

    def test_trim_to_field_no_sources_raises_error(self):
        pass
        
    def test_trim_to_field_naxis_4(self):
        
        # Mock naxis=4 data array
        x, y = self.northimage.data.shape
        self.northimage.data = np.reshape(self.northimage.data, (1, 1, x, y))

        racs = self.northimage.get_catalogue('RACS')
        self.assertEqual(len(racs), 50)

