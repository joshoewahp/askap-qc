#!/usr/bin/env python

import logging
import os
import re
import astropy.units as u
import numpy as np
import pandas as pd
from astropy import wcs
from astropy.coordinates import SkyCoord, Angle
from astropy.io import fits, votable
from astroquery.vizier import Vizier
from collections import namedtuple
from pathlib import Path


Filepair = namedtuple('Filepair', 'image selavy')


class Region:

    def __init__(self, regions, band='low'):
        self.band: str = band
        self._register_fields()

        self.name: str = '#' + '-'.join([r for r in regions])
        self.fields: list = [f for region in regions for f in self.regions[region]]

    def __repr__(self):
        return f'Region {self.footprint}-{self.name}'

    def _register_fields(self):

        if self.band == 'low':
            self.regions = {
                '1': ['2004+00', '2004-06', '2028+00', '2028-06', '2053+00', '2053-06', '2118+00',
                      '2118-06', '2143+00', '2143-06', '2208+00', '2208-06', '2257+00', '2257-06',
                      '2322+00', '2322-06', '2347+00', '2347-06', '2233+00', '2233-06', '0012+00',
                      '0012-06', '0037+00', '0037-06', '0102+00', '0102-06', '0126+00', '0126-06',
                      '0151+00', '0151-06', '0216+00', '0216-06', '0241+00', '0241-06', '0306+00',
                      '0306-06', '0331+00', '0331-06', '0355+00', '0355-06',],
                '2': ['0918+00', '0918-06', '0943+00', '0943-06', '1008+00', '1008-06', '1033+00',
                      '1033-06', '1057+00', '1057-06', '1122+00', '1122-06', '1147+00', '1147-06',
                      '1212+00', '1212-06', '1237+00', '1237-06', '1302+00', '1302-06', '1326+00',
                      '1326-06', '1351+00', '1351-06', '1416+00', '1416-06', '1441+00', '1441-06',],
                '3': ['0304-50', '0310-56', '0318-62', '0320-43', '0341-50', '0352-56', '0354-43',
                      '0408-62', '0418-50', '0427-43', '0435-56', '0455-50', '0457-62', '0501-43',
                      '0517-56', '0530-68', '0532-50', '0534-43', '0547-62', '0559-56',],
                '4': ['2005-43', '2007-56', '2018-50', '2039-43', '2041-62', '2049-56', '2055-50',
                      '2112-43', '2131-56', '2131-62', '2132-50', '2146-43', '2209-50', '2214-56',
                      '2219-43', '2220-62', '2246-50', '2253-43', '2256-56',],
                '5': ['1724-31', '1739-25', '1752-31', '1753-18', '1806-25',],
                '6': ['0127-73'],
                }

        elif self.band =='mid':
            self.regions = {
                '1': ['0021+00', '0021-04', '0042+00', '0042-04', '0104+00', '0104-04', '0125+00',
                      '0125-04', '0147+00', '0147-04', '0208+00', '0208-04', '0230+00', '0230-04',
                      '0251+00', '0251-04', '0313+00', '0313-04', '0334+00', '0334-04', '0352-64',
                      '0356+00', '0356-04', '2108+00', '2108-04', '2129+00', '2129-04', '2003+00',
                      '2003-04', '2046+00', '2046-04', '2255+00', '2255-04', '2306-55', '2234+00',
                      '2234-04', '2317+00', '2317-04', '2338+00', '2338-04', '2359+00', '2359-04' 
                      '2025+00', '2025-04'],
                '2': [],
                '3': ['0438-64', '0504-69', '0525-64', '0559-69'],
                '4': ['2004-41', '2006-55', '2014-46', '2027-51', '2032-41', '2034-60', '2042-55',
                      '2044-46', '2054-64', '2059-41', '2059-51', '2114-46', '2115-60', '2118-55',
                      '2127-41', '2132-51', '2140-64', '2144-46', '2151+00', '2151-04', '2154-55',
                      '2155-41', '2156-60', '2205-51', '2212+00', '2212-04', '2214-46', '2223-41',
                      '2227-64', '2230-55', '2237-60', '2238-51', '2244-46', '2250-41'],
                '5': ['1724-28', '1731-23', '1735-32', '1748-18', '1748-28', '1754-23', '1800-32',
                      '1812-28'],
                '6': ['0113-72']
            }

        
class Epoch:

    def __init__(self, rootpath, tiletype, stokes, regions, band):
        self.rootpath = Path(rootpath)
        self.tiletype = tiletype
        self.stokes = stokes
        self.region = Region(regions, band) if regions else None
        self.path = self.rootpath / tiletype
        self.name = self.rootpath.parts[-1]
        self.logger = logging.getLogger(__name__ + f' - {self.name}')
        self._parse_files()


    def __repr__(self):
        return f'<{self.name}-{self.stokes}-{self.tiletype}>'
        
    def _parse_files(self):

        image_files = list(self.path.glob(f'STOKES{self.stokes}_IMAGES/*{self.stokes}.fits'))
        selavy_files = list(self.path.glob(f'STOKES{self.stokes}_SELAVY/*components.txt'))

        if self.region:
            self.image_files = [f for field in self.region.fields for f in image_files if field in str(f)]
            self.selavy_files = [f for field in self.region.fields for f in selavy_files if field in str(f)]
        else:
            self.image_files = image_files
            self.selavy_files = selavy_files
            
        num_images = len(self.image_files)
        num_selavy = len(self.selavy_files)    

        self.logger.info(f"{num_images:>4} images and {num_selavy:>4} selavy files in epoch {self.name}.")

        # Regex pattern to select field name (e.g. 0012+00A)
        pattern = re.compile(r'\S*(\d{4}[-+]\d{2}[AB])\S*')
        self.files = [Filepair(im, sel) for im in self.image_files for sel in self.selavy_files if
                      pattern.sub(r'\1', str(sel)) in str(im)]
        self.num_images = len(self.files)
        if self.num_images < num_images:
            self.logger.warning(f"Only {self.num_images}/{num_images} images have matching selavy file.")


class Image:

    def __new__(cls, filepair, load_data=True, **kwargs):
        if not (os.path.isfile(filepair.image) or os.path.islink(filepair.image)):
            raise ValueError("Dataset does not seem to exist, check path!")
        else:
            return super().__new__(cls)

    def __init__(self, filepair, load_data=True, **kwargs):
        self.imagepath = filepair.image
        self.selavypath = filepair.selavy
        pattern = re.compile(r'\S*(\d{4}[+-]\d{2}[AB])\S*')
        self.name = pattern.sub(r'\1', str(self.imagepath))
        self.refcat = kwargs.get('refcat')
        self.kwargs = kwargs
        
        self.logger = logging.getLogger(__name__ + f' - {self.name}')

        self._parse_name()
        self._load(load_data)

    def __repr__(self):
        return "<{} {} Stokes{}>".format(self.epoch, self.fieldname, self.polarisation)

    def _parse_name(self):
        pattern = re.compile(r'^(\S*_\d{4}[+-]\d{2}[AB]).(EPOCH\d{2}x*).([IV]).fits')
        self.fieldname = pattern.sub(r'\1', self.name)
        self.epoch = pattern.sub(r'\2', self.name)
        self.polarisation = pattern.sub(r'\3', self.name)

    def _load(self, load_data):
        with fits.open(self.imagepath) as hdul:
            self.header = hdul[0].header
            self.data = hdul[0].data if load_data else None

        self.wcs = wcs.WCS(self.header, naxis=2)
        self.sbid = self.header.get('SBID')
        self.frequency = self.header.get('CRVAL3') if self.header.get('CTYPE3') == 'FREQ' else None
        self.frequency = self.header.get('RESTFREQ', self.frequency)
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
        centre, edge, corner = self._get_field_positions()
        self.cr_ra, self.cr_dec = centre[0][0], centre[0][1]
        self.fieldcentre = SkyCoord(ra=self.cr_ra, dec=self.cr_dec, unit=u.deg)
        self.edge = SkyCoord(ra=edge[0][0], dec=edge[0][1], unit=u.deg)
        self.corner = SkyCoord(ra=corner[0][0], dec=corner[0][1], unit=u.deg)
        self.edgeradius = self.fieldcentre.separation(self.edge)
        self.cornerradius = self.fieldcentre.separation(self.corner)

    def get_catalogue(self, catalogue, boundary_value='nan', **kwargs):
        """Query Vizier for catalogue sources within field coverage."""
        cat_ids = {"SUMSS": "VIII/81B/sumss212",
                   "NVSS": "VIII/65/nvss",
                   "MGPS2": "VIII/82/mgpscat",
                   "ICRF": "I/323/icrf2"}

        if catalogue == 'REF':
            
            sources = self.refcat.copy()
            sources = self._trim_to_field(sources)
            
            return sources
            
            
        elif catalogue == 'ASKAP':

            # Handle loading of multiple source file formats
            if self.selavypath.suffix in ['.xml', '.vot']:
                sources = Table.read(
                    self.selavypath, format="votable", use_names_over_ids=True
                ).to_pandas()
            elif self.selavypath.suffix == '.csv':
                # CSVs from CASDA have all lowercase column names
                sources = pd.read_csv(self.selavypath).rename(
                    columns={"spectral_index_from_tt": "spectral_index_from_TT"}
                )
            else:
                sources = pd.read_fwf(self.selavypath, skiprows=[1])

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

        v = Vizier(columns=["_r", "_RAJ2000", "_DEJ2000", "**"], row_limit=-1)

        result = v.query_region(self.fieldcentre,
                                radius=self.cornerradius * 1.1,
                                catalog=search_cat)
        try:
            assert len(result.keys()) > 0, "No {} sources located in this field.".format(catalogue)
        except AssertionError:
            raise

        df_result = pd.concat((result[cat].to_pandas() for cat in result.keys()), sort=False)
        matches = self._clean_matches(catalogue, df_result)
        self.matches = self._trim_to_field(matches)
        assert len(self.matches) > 0, "No {} sources located in this field.".format(catalogue)

        return self.matches

    def _clean_matches(self, catalogue, df_result):
        """
        Create uniform column names and assign scalar values for
        missing information (e.g. local rms, peak flux)
        """
        if catalogue == 'NVSS':
            df_result.rename(columns={'S1.4': 'flux_int', 'e_S1.4': 'flux_int_err',
                                      'MajAxis': 'maj_axis', 'MinAxis': 'min_axis', 'PA': 'pos_ang',
                                      '_RAJ2000': 'ra', '_DEJ2000': 'dec',
                                      'e_RAJ2000': 'ra_err', 'e_DEJ2000': 'dec_err'},
                             inplace=True)

            # Convert RA errors from hms seconds -> degrees
            df_result['ra_err'] = Angle((0, 0, df_result['ra_err']),
                                        unit=u.hour).degree
            # No peak flux provided, replace with NaN
            df_result['flux_peak'] = np.nan
            df_result['flux_peak_err'] = np.nan
            df_result['rms_image'] = 0.45
            df_result['pos_ang'].fillna(0.0, inplace=True)

        elif catalogue == 'SUMSS':
            df_result.rename(columns={'St': 'flux_int', 'e_St': 'flux_int_err',
                                      'MajAxis': 'maj_axis', 'MinAxis': 'min_axis', 'PA': 'pos_ang',
                                      '_RAJ2000': 'ra', '_DEJ2000': 'dec',
                                      'e_RAJ2000': 'ra_err', 'e_DEJ2000': 'dec_err'},
                             inplace=True)

            # No peak flux provided, replace with NaN
            df_result['flux_peak'] = np.nan
            df_result['flux_peak_err'] = np.nan
            # Convert RA errors from arcseconds -> degrees
            df_result['ra_err'] = Angle(df_result['ra_err'], unit=u.arcsec).degree
            # Dec dependent RMS according to paper
            df_result['rms_image'] = df_result.apply(lambda x: 2. if x.dec > -50 else 1.2, axis=1)

        elif catalogue == 'ICRF':
            df_result.rename(columns={'_RAJ2000': 'ra', '_DEJ2000': 'dec',
                                      'e_RAJ2000': 'ra_err', 'e_DEJ2000': 'dec_err'},
                             inplace=True)

            # Convert RA errors from hms seconds -> degrees
            df_result['ra_err'] = Angle((0, 0, df_result['ra_err']),
                                         unit=u.hour).degree
            # No fluxes / beam parameters provided, replace with NaN
            for col in ['flux_int', 'flux_int_err', 'flux_peak', 'flux_peak_err',
                        'maj_axis', 'min_axis', 'pos_ang', 'rms_image']:
                df_result[col] = np.nan

        # Convert Dec errors from arcseconds -> degrees
        df_result['dec_err'] = Angle(df_result['dec_err'], unit=u.arcsec).degree
        df_result = df_result[['_r', 'ra', 'dec', 'ra_err', 'dec_err',
                               'maj_axis', 'min_axis', 'pos_ang',
                               'flux_int', 'flux_int_err',
                               'flux_peak', 'flux_peak_err', 'rms_image']].copy()

        return df_result

    def _get_field_positions(self):
        """Calculate coordinates of field centre, horizontal edge, and corner."""
        centre = np.array([[self.size_x / 2., self.size_y / 2.]])
        if self.size_x >= self.size_y:
            edge = np.array([[self.size_x, self.size_y / 2.]])
        else:
            edge = np.array([[self.size_x / 2., self.size_y]])
        corner = np.array([[self.size_x, self.size_y]])

        centre = self.wcs.wcs_pix2world(centre, 1)
        edge = self.wcs.wcs_pix2world(edge, 1)
        corner = self.wcs.wcs_pix2world(corner, 1)

        return centre, edge, corner

    def _trim_to_field(self, matches):
        """Remove any matches with coordinates outside the NaN boundary."""

        # Rough RA / Dec cuts to +- 10 deg from fieldd centre for efficiency
        dist = 10
        matches = matches[(matches.ra > self.cr_ra - dist) &
                          (matches.ra < self.cr_ra + dist) &
                          (matches.dec > self.cr_dec - dist) &
                          (matches.dec < self.cr_dec + dist)].copy()
        
        raw_coords = [[row.ra, row.dec] for i, row in matches.iterrows()]
        pixel_coords = self.wcs.wcs_world2pix(raw_coords, 1)

        matches.loc[:, 'pix_x'] = pixel_coords[:, 0].astype(np.int)
        matches.loc[:, 'pix_y'] = pixel_coords[:, 1].astype(np.int)

        # Trim to image pixel size so data can be indexed
        matches = matches[(matches.pix_x.between(1, self.size_x - 1)) &
                          (matches.pix_y.between(1, self.size_y - 1))]

        # Remove NaN values at image boundary (x,y reversed in data array)
        if len(self.data.shape) > 2:
            matches = matches[~np.isnan(self.data[0, 0, matches.pix_y, matches.pix_x])]
        else:
            matches = matches[~np.isnan(self.data[matches.pix_y, matches.pix_x])]

        return matches.drop(columns=['pix_x', 'pix_y'])

