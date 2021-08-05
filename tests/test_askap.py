import numpy as np
from catalog import ReferenceCatalog
from askap import Filepair, Epoch, Region, Image
from unittest import TestCase
from pathlib import Path


class EpochTest(TestCase):

    def setUp(self) -> None:
        self.epoch8path = Path('data/mockdata/EPOCH08/')
        return super().setUp()

    def test_repr(self) -> None:
        epoch = Epoch(self.epoch8path, tiletype='COMBINED', stokes='I', regions=None, band='low')
        self.assertEqual(repr(epoch),
                         "Epoch(Path('data/mockdata/EPOCH08'), tiletype='COMBINED', stokes='I', regions=None, band='low')")
        self.assertEqual(str(epoch), "<EPOCH08-I-COMBINED>")
        
        
    def test_bad_tiletype_arg(self) -> None:
        self.assertRaises(ValueError,
                          Epoch,
                          self.epoch8path, tiletype='TILE', stokes='V', regions=('3', '4'), band='low')

    def test_combined_low_I_no_region(self) -> None:
        epoch = Epoch(self.epoch8path, tiletype='COMBINED', stokes='I', regions=None, band='low')
        self.assertEqual(epoch.num_files, 112) 

    def test_combined_low_I_region(self) -> None:
        epoch = Epoch(self.epoch8path, tiletype='COMBINED', stokes='I', regions=('3', '4'), band='low')
        self.assertEqual(epoch.num_files, 39) 
        
    def test_tiles_low_I_no_region(self) -> None:
        epoch = Epoch(self.epoch8path, tiletype='TILES', stokes='I', regions=None, band='low')
        self.assertEqual(epoch.num_files, 112) 

    def test_tiles_low_I_region(self) -> None:
        epoch = Epoch(self.epoch8path, tiletype='TILES', stokes='I', regions=('3', '4'), band='low')
        self.assertEqual(epoch.num_files, 39) 

    def test_combined_low_V_no_region(self) -> None:
        epoch = Epoch(self.epoch8path, tiletype='COMBINED', stokes='V', regions=None, band='low')
        self.assertEqual(epoch.num_files, 224) 

    def test_combined_low_V_region(self) -> None:
        epoch = Epoch(self.epoch8path, tiletype='COMBINED', stokes='V', regions=('3', '4'), band='low')
        self.assertEqual(epoch.num_files, 78) 

    def test_tiles_low_V_no_region(self) -> None:
        self.assertRaises(NotImplementedError,
                          Epoch,
                          self.epoch8path, tiletype='TILES', stokes='V', regions=None, band='low')

    def test_tiles_low_V_region(self) -> None:
        self.assertRaises(NotImplementedError,
                          Epoch,
                          self.epoch8path, tiletype='TILES', stokes='V', regions=('3', '4'), band='low')


class RegionTest(TestCase):

    def setUp(self) -> None:
        self.regions = ['1', '2', '3', '4', '5', '6']
        return super().setUp()

    def test_repr(self) -> None:
        reg = Region(regions=('3', '4'), band='low')
        self.assertEqual(repr(reg), "Region(regions=('3', '4'), band='low')")
        self.assertEqual(str(reg), "<Region: low 3-4>")

    def test_low_band(self) -> None:
        nfields = [40, 28, 20, 19, 5, 1]
        for region, number in zip(self.regions, nfields):
            reg = Region(regions=region, band='low')
            self.assertEqual(len(reg.fields), number)
            
    def test_mid_band(self) -> None:
        nfields = [43, 0, 4, 34, 8, 1]
        for region, number in zip(self.regions, nfields):
            reg = Region(regions=region, band='mid')
            self.assertEqual(len(reg.fields), number)

    def test_high_band(self) -> None:
        nfields = [0, 0, 0, 0, 0, 0]
        for region, _ in zip(self.regions, nfields):
            self.assertRaises(NotImplementedError, Region, regions=region, band='high')

    def test_bad_band_argument(self) -> None:
        self.assertRaises(ValueError, Region, regions='1', band='hi')


class ImageTest(TestCase):

    @classmethod
    def setUpClass(self) -> None:
        rootpath = Path('data/testdata/EPOCH08/COMBINED/')

        imagepath_south = rootpath / 'STOKESI_IMAGES/VAST_0127-73A.EPOCH08.I.fits'
        selavypath_south = rootpath / 'STOKESI_SELAVY/VAST_0127-73A.EPOCH08.I.selavy.components.txt'
        imagepath_north = rootpath / 'STOKESI_IMAGES/VAST_0012+00A.EPOCH08.I.fits'
        selavypath_north = rootpath / 'STOKESI_SELAVY/VAST_0012+00A.EPOCH08.I.selavy.components.txt'

        self.refcat = ReferenceCatalog('data/testdata/RACS_catalogue_test.fits')
        self.southfiles = Filepair(image=imagepath_south, selavy=selavypath_south)
        self.northfiles = Filepair(image=imagepath_north, selavy=selavypath_north)

        return super().setUpClass()

    def setUp(self) -> None:
        self.northimage = Image(self.northfiles, refcat=self.refcat.sources)

        return super().setUp()

    def check_column_names(self, catalogue) -> None:
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
    
    def check_header_vals(self, image) -> None:
        """Check value of header keywords are set correctly."""

        self.assertEqual(image.sbid, None)
        self.assertEqual(image.bmaj, 0.00364940923141675)
        self.assertEqual(image.bmin, 0.00356381661788453)
        self.assertEqual(image.bpa, 5.28573604455489)
        self.assertEqual(image.size_x, 13700)
        self.assertEqual(image.size_y, 13690)
        self.assertEqual(image.bcr_ra, 3.102094583333)
        self.assertEqual(image.bcr_dec, 0.003423722222222)
        self.assertEqual(image.cr_ra, 3.102441805555726)
        self.assertEqual(image.cr_dec, 0.0030764999999284407)
        self.assertEqual(image.edgeradius.deg, 4.758940991409037)
        self.assertEqual(image.cornerradius.deg, 6.740414010519901)
    
    def test_repr(self) -> None:
        self.assertEqual(repr(self.northimage), f"<Image: EPOCH08 0012+00A StokesI>")
        self.assertEqual(str(self.northimage), f"<Image: EPOCH08 0012+00A StokesI>")

    def test_init(self) -> None:
        self.assertEqual(self.northimage.filepair, self.northfiles)
        self.assertEqual(self.northimage.imagepath, self.northfiles.image)
        self.assertEqual(self.northimage.selavypath, self.northfiles.selavy)
        self.assertEqual(self.northimage.refcat.shape, self.refcat.sources.shape)

    def test_load(self) -> None:
        self.check_header_vals(self.northimage)
        self.assertEqual(self.northimage.data.shape, (13690, 13700))

    def test_load_no_data(self) -> None:
        image = Image(self.northfiles, refcat=self.refcat.sources, load_data=False)
        self.check_header_vals(image)
        self.assertEqual(image.data, None)
        
    def test_find_nearest_edge_axes_swapped(self) -> None:
        image = Image(self.northfiles, refcat=self.refcat.sources, load_data=False)

        # Mock swapped image dimensions
        image.data = self.northimage.data.T
        image.size_x, image.size_y = self.northimage.size_y, self.northimage.size_x

        image._find_nearest_edge()
        self.assertEqual(image.edgeradius.deg, 4.758940991409037)
    
    def test_parse_name(self) -> None:
        self.assertEqual(self.northimage.fieldname, '0012+00A')
        self.assertEqual(self.northimage.epoch, 'EPOCH08')
        self.assertEqual(self.northimage.stokes, 'I')

    def test_parse_nonmatching_name(self) -> None:
        rootpath = Path('data/testdata/EPOCH05x/COMBINED/')
        imagepath = rootpath / 'STOKESI_IMAGES/VAST_0012+00A.EPOCH05x.I.conv.fits'
        selavypath = rootpath / 'STOKESI_SELAVY/VAST_0012+00A.EPOCH05x.I.selavy.components.txt'
        files = Filepair(image=imagepath, selavy=selavypath)

        image = Image(files, refcat=self.refcat.sources)

        self.assertEqual(image.fieldname, '0012+00A')
        self.assertEqual(image.epoch, None)
        self.assertEqual(image.stokes, None)
        
    def test_get_frequency_with_restfreq(self) -> None:
        freq = self.northimage._get_frequency()
        self.assertEqual(freq, 887.491e6)

    def test_get_frequency_with_crval3(self) -> None:
        restfreq = self.northimage.header.pop('RESTFREQ')
        self.northimage.header['CRVAL3'] = restfreq
        self.northimage.header['CTYPE3'] = 'FREQ'

        freq = self.northimage._get_frequency()
        self.assertEqual(freq, 887.491e6)

    def test_get_frequency_without_restfreq_or_crval3(self) -> None:
        self.northimage.header.pop('RESTFREQ')
        self.assertRaises(KeyError, self.northimage._get_frequency)

    def test_get_catalogue_askap(self) -> None:
        askap = self.northimage.get_catalogue('ASKAP')

        self.assertEqual(len(askap), 7976)
        self.check_column_names(askap)
        
    def test_get_catalogue_sumss_south(self) -> None:
        southimage = Image(self.southfiles, refcat=self.refcat.sources)
        sumss = southimage.get_catalogue('SUMSS')

        self.assertEqual(len(sumss), 1498)
        self.check_column_names(sumss)

    def test_get_catalogue_sumss_north(self) -> None:
        self.assertRaises(AssertionError, self.northimage.get_catalogue, 'SUMSS')

    def test_get_catalogue_nvss_south(self) -> None:
        southimage = Image(self.southfiles, refcat=self.refcat.sources)
        self.assertRaises(AssertionError, southimage.get_catalogue, 'NVSS')

    def test_get_catalogue_nvss_north(self) -> None:
        nvss = self.northimage.get_catalogue('NVSS')

        self.assertEqual(len(nvss), 3869)
        self.check_column_names(nvss)
        
    def test_get_catalogue_icrf(self) -> None:
        icrf = self.northimage.get_catalogue('ICRF')

        self.assertEqual(len(icrf), 6)
        self.check_column_names(icrf)

    def test_get_catalogue_racs(self) -> None:
        racs = self.northimage.get_catalogue('RACS')

        self.assertEqual(len(racs), 2268)
        self.check_column_names(racs)

    def test_trim_to_field_naxis_4(self) -> None:
        
        # Mock naxis=4 data array
        x, y = self.northimage.data.shape
        self.northimage.data = np.reshape(self.northimage.data, (1, 1, x, y))

        racs = self.northimage.get_catalogue('RACS')
        self.assertEqual(len(racs), 2268)

