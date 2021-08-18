import logging
import re
import astropy.units as u
import numpy as np
import pandas as pd
from astropy import wcs
from astropy.coordinates import SkyCoord, Angle
from astropy.io import fits
from astroquery.vizier import Vizier
from dataclasses import dataclass
from fileio import load_selavy_file
from pathlib import Path
from region import LowBandRegion, MidBandRegion

logger = logging.getLogger(__name__)


@dataclass
class Filepair:
    """Matched pair of a FITS image and associated selavy catalogue."""

    image: Path
    selavy: Path


class Region:
    """Simple API to generate VAST regions in a given frequency band."""
    
    def __new__(cls, regions: tuple[str], band: str = 'low'):

        if band == 'low':
            return LowBandRegion(regions)
        if band == 'mid':
            return MidBandRegion(regions)
        if band == 'high':
            raise NotImplementedError("High band not yet available")

        raise ValueError("Must pass band value of 'low', 'mid', or 'high'")
    

@dataclass
class Epoch:
    """Representation of image/selavy Filepairs within a VAST epoch."""

    path: Path
    region: Region
    tiletype: str
    stokes: str
    band: str

    def __post_init__(self):
        self.name = self.path.name
        self.path = self.path / self.tiletype
        self._parse_files()

    def _parse_files(self):

        if self.tiletype == 'TILES' and self.stokes == 'V':
            raise NotImplementedError("Stokes V data unavailable for TILES")

        if self.tiletype == 'COMBINED':
            image_files = list(self.path.glob(f'STOKES{self.stokes}_IMAGES/*{self.stokes}.fits'))
            selavy_files = list(self.path.glob(f'STOKES{self.stokes}_SELAVY/*components.txt'))
        elif self.tiletype == 'TILES':
            image_files = list(self.path.glob(f'STOKES{self.stokes}_IMAGES/*{self.stokes.lower()}*restored.fits'))
            selavy_files = list(self.path.glob(f'STOKES{self.stokes}_SELAVY/*components.txt'))
        else:
            raise ValueError("Must pass tiletype value of COMBINED or TILES")

        if self.region:
            image_files = [f for field in self.region for f in image_files if field in str(f)]
            selavy_files = [f for field in self.region for f in selavy_files if field in str(f)]
            
        # Regex pattern to select field name (e.g. 0012+00A)
        pattern = re.compile(r'\S*(\d{4}[-+]\d{2}[AB])\S*')
        self.files = [Filepair(im, sel) for im in image_files for sel in selavy_files if
                      pattern.sub(r'\1', str(sel)) in str(im)]
        self.num_files = len(self.files)


@dataclass
class Image:
    """Representation of an ASKAP image."""

    filepair: Filepair
    refcat: 'ReferenceCatalog'
    load_data: bool = True

    def __post_init__(self):
        self.imagepath = self.filepair.image
        self.selavypath = self.filepair.selavy
        self.fieldname = self.imagepath.name

        self._parse_name()

        self._load(self.load_data)

    def __repr__(self):
        return f"<Image: {self.imagepath.parts[-1]!r}>"
        
    def _parse_name(self):
        pattern = re.compile(r'^\S*_(\d{4}[+-]\d{2}[AB]).(EPOCH\d{2}x*).([IV]).fits')
        self.fieldname = pattern.sub(r'\1', str(self.imagepath))
        self.epoch = pattern.sub(r'\2', str(self.imagepath))
        self.stokes = pattern.sub(r'\3', str(self.imagepath))

        # Use less strict regex if unable to parse fieldname
        if self.fieldname == str(self.imagepath):
            pattern = re.compile(r'\S*(\d{4}[+-]\d{2}[AB])\S*')
            self.fieldname = pattern.sub(r'\1', str(self.imagepath))
            self.epoch = None
            self.stokes = None

    def _get_frequency(self):
        """Read observing frequency from one of multiple FITS header keywords"""

        frequency = self.header.get('RESTFREQ')
        if frequency:
            return frequency
        elif self.header.get('CTYPE3') == 'FREQ':
            return self.header.get('CRVAL3')
        else:
            raise KeyError('No RESTFREQ or CRVAL3 keywords found in header')

    def _load(self, load_data: bool):
        with fits.open(self.imagepath) as hdul:
            self.header = hdul[0].header
            self.data = hdul[0].data if load_data else None

        self.wcs = wcs.WCS(self.header, naxis=2)
        self.sbid = self.header.get('SBID')
        self.frequency = self._get_frequency()
        self.bmaj = self.header.get('BMAJ')
        self.bmin = self.header.get('BMIN')
        self.bpa = self.header.get('BPA')

        # Array dimensions in pixels
        self.size_x = self.header.get('NAXIS1')
        self.size_y = self.header.get('NAXIS2')

        # World coordinates of beam 0
        self.bcr_ra = self.header.get('CRVAL1')
        self.bcr_dec = self.header.get('CRVAL2')

        # Coordinates of field centre, nearest edge, and top right corner.
        self._set_field_positions()

    def get_catalogue(self, catalogue: str) -> pd.DataFrame:
        """Query Vizier for catalogue sources within field coverage."""
        cat_ids = {"SUMSS": "VIII/81B/sumss212",
                   "NVSS": "VIII/65/nvss",
                   "MGPS2": "VIII/82/mgpscat",
                   "ICRF": "I/323/icrf2"}

        if catalogue in ['RACS', 'REF']:
            
            sources = self.refcat.copy()
            sources = self._trim_to_field(sources)
            
            return sources
            
            
        elif catalogue == 'ASKAP':

            sources = load_selavy_file(self.selavypath)

            sources.rename(columns={'ra_deg_cont': 'ra', 'dec_deg_cont': 'dec',
                                    'ra_deg_cont_err': 'ra_err', 'dec_deg_cont_err': 'dec_err'},
                           inplace=True)
            sources = sources[['island_id', 'component_id', 'component_name',
                               'ra', 'dec', 'ra_err', 'dec_err', 'maj_axis', 'min_axis', 'pos_ang', 
                               'flux_peak', 'flux_peak_err', 'flux_int', 'flux_int_err', 'rms_image']]

            return sources
            
        elif catalogue == 'SUMSS':
            search_cat = [cat_ids['SUMSS'], cat_ids['MGPS2']]
        else:
            search_cat = cat_ids.get(catalogue)

        assert search_cat is not None, "{} not recognised as Vizier catalogue.".format(catalogue)

        v = Vizier(columns=["_r", "_RAJ2000", "_DEJ2000", "PA", "*"], row_limit=-1)

        result = v.query_region(self.fieldcentre,
                                radius=self.cornerradius * 1.1,
                                catalog=search_cat)

        try:
            assert len(result.keys()) > 0, "No {} sources located in this field.".format(catalogue)
        except AssertionError:
            raise

        vizier_df = pd.concat((result[cat].to_pandas() for cat in result.keys()), sort=False)
        matches = self._clean_matches(catalogue, vizier_df)
        matches = self._trim_to_field(matches)
        assert len(matches) > 0, "No {} sources located in this field.".format(catalogue)

        return matches

    def _clean_matches(self, catalogue: str, vizier_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create uniform column names and assign scalar values for
        missing information (e.g. local rms, peak flux)
        """

        if catalogue == 'NVSS':
            vizier_df.rename(columns={'S1.4': 'flux_int', 'e_S1.4': 'flux_int_err',
                                      'MajAxis': 'maj_axis', 'MinAxis': 'min_axis', 'PA': 'pos_ang',
                                      '_RAJ2000': 'ra', '_DEJ2000': 'dec',
                                      'e_RAJ2000': 'ra_err', 'e_DEJ2000': 'dec_err'},
                             inplace=True)

            # Convert RA errors from hms seconds -> degrees
            vizier_df['ra_err'] = Angle((0, 0, vizier_df['ra_err']), unit=u.hour).degree

            # No peak flux, rms, or beam errors provided
            vizier_df['flux_peak'] = np.nan
            vizier_df['flux_peak_err'] = np.nan
            vizier_df['rms_image'] = 0.45
            vizier_df['pos_ang'].fillna(0.0, inplace=True)

        elif catalogue == 'SUMSS':
            vizier_df.rename(columns={'St': 'flux_int', 'e_St': 'flux_int_err',
                                      'MajAxis': 'maj_axis', 'MinAxis': 'min_axis', 'PA': 'pos_ang',
                                      '_RAJ2000': 'ra', '_DEJ2000': 'dec',
                                      'e_RAJ2000': 'ra_err', 'e_DEJ2000': 'dec_err'},
                             inplace=True)

            # No peak flux or beam errors provided
            vizier_df['flux_peak'] = np.nan
            vizier_df['flux_peak_err'] = np.nan

            # Convert RA errors from arcseconds -> degrees
            vizier_df['ra_err'] = Angle(vizier_df['ra_err'], unit=u.arcsec).degree

            # Dec dependent RMS according to Mauch et al. (2)
            vizier_df['rms_image'] = vizier_df.apply(lambda x: 2. if x.dec > -50 else 1.2, axis=1)

        elif catalogue == 'ICRF':
            vizier_df.rename(columns={'_RAJ2000': 'ra', '_DEJ2000': 'dec',
                                      'e_RAJ2000': 'ra_err', 'e_DEJ2000': 'dec_err'},
                             inplace=True)

            # Convert RA errors from hms seconds -> degrees
            vizier_df['ra_err'] = Angle((0, 0, vizier_df['ra_err']), unit=u.hour).degree

            # No fluxes / beam parameters provided, replace with NaN
            for col in ['flux_int', 'flux_int_err', 'flux_peak', 'flux_peak_err',
                        'maj_axis', 'min_axis', 'pos_ang', 'rms_image']:
                vizier_df[col] = np.nan

        # Convert Dec errors from arcseconds -> degrees
        vizier_df['dec_err'] = Angle(vizier_df['dec_err'], unit=u.arcsec).degree
        vizier_df = vizier_df[['_r', 'ra', 'dec', 'ra_err', 'dec_err',
                               'maj_axis', 'min_axis', 'pos_ang',
                               'flux_int', 'flux_int_err',
                               'flux_peak', 'flux_peak_err', 'rms_image']].copy()

        return vizier_df

    def _find_nearest_edge(self):
        """Check pixel dimensions to select pixel coordinates of nearest edge"""

        if self.size_x <= self.size_y:
            edge_pixel_coords = np.array([[self.size_x, self.size_y / 2.]])
        else:
            edge_pixel_coords = np.array([[self.size_x / 2., self.size_y]])

        return edge_pixel_coords
    
    def _set_field_positions(self):
        """Calculate coordinates of field centre, horizontal edge, and corner."""

        edge_pixel_coords = self._find_nearest_edge()
        edge = self.wcs.wcs_pix2world(edge_pixel_coords, 1)
        centre = np.array([[self.size_x / 2., self.size_y / 2.]])
        centre = self.wcs.wcs_pix2world(centre, 1)
        corner = np.array([[self.size_x, self.size_y]])
        corner = self.wcs.wcs_pix2world(corner, 1)
        
        self.cr_ra, self.cr_dec = centre[0][0], centre[0][1]
        self.fieldcentre = SkyCoord(ra=self.cr_ra, dec=self.cr_dec, unit=u.deg)
        edge_coord = SkyCoord(ra=edge[0][0], dec=edge[0][1], unit=u.deg)
        corner_coord = SkyCoord(ra=corner[0][0], dec=corner[0][1], unit=u.deg)
        self.edgeradius = self.fieldcentre.separation(edge_coord)
        self.cornerradius = self.fieldcentre.separation(corner_coord)

    def _trim_to_field(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Remove any matches with coordinates outside the NaN boundary."""

        # Rough RA / Dec cuts to +- 10 deg from field centre for efficiency
        dist = 10
        matches = matches[(matches.ra > self.cr_ra - dist) &
                          (matches.ra < self.cr_ra + dist) &
                          (matches.dec > self.cr_dec - dist) &
                          (matches.dec < self.cr_dec + dist)].copy()

        # Trim to image pixel size so data can be indexed
        raw_coords = [[row.ra, row.dec] for _, row in matches.iterrows()]
        pixel_coords = self.wcs.wcs_world2pix(raw_coords, 1)

        matches.loc[:, 'pix_x'] = pixel_coords[:, 0].astype(int)
        matches.loc[:, 'pix_y'] = pixel_coords[:, 1].astype(int)

        matches = matches[(matches.pix_x.between(1, self.size_x - 1)) &
                          (matches.pix_y.between(1, self.size_y - 1))]

        # Remove NaN values at image boundary (x,y transposed in data array)
        if len(self.data.shape) > 2:
            matches = matches[~np.isnan(self.data[0, 0, matches.pix_y, matches.pix_x])]
        else:
            matches = matches[~np.isnan(self.data[matches.pix_y, matches.pix_x])]

        return matches.drop(columns=['pix_x', 'pix_y'])

