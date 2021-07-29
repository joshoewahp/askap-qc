#!/usr/env/bin python

import os
import logging
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table import Table
from askap import Image
from fileio import load_selavy_file
from pathlib import Path


logger = logging.getLogger(__name__)


class ReferenceCatalog:
    """Source catalogue as reference for flux and astrometry comparisons

    catpath should point either to the published RACS catalogue
    as an astropy table or a raw selavy component catalogue
    """

    def __init__(self, catpath: Path):
        self._set_reference_catalogue(catpath)

    def _set_reference_catalogue(self, catpath):

        if 'RACS' in catpath:

            # Get RACS catalogue as base
            self.name = 'racs'

            refcat = Table.read(catpath).to_pandas()
            columns = {
                'N_Gaus': 'N_Gaus',
                'RA': 'ra',
                'Dec': 'dec',
                'Total_flux_Component': 'flux_int',
                'E_Total_flux_Component': 'flux_int_err',
                'Peak_flux': 'flux_peak',
                'E_Peak_flux': 'flux_peak_err',
                'Maj': 'maj_axis',
                'E_Maj': 'maj_axis_err',
                'Min': 'min_axis',
                'E_Min': 'min_axis_err',
                'PA': 'pos_ang',
                'E_PA': 'pos_ang_err',
                'Separation_Tile_Centre': 'field_centre_dist',
                'Noise': 'rms_image'
            }
            refcat = refcat.rename(columns=columns)[columns.values()]
        else:

            self.name = 'ref'
            refcat = load_selavy_file(catpath)
            refcat = refcat.rename(columns={'ra_deg_cont': 'ra',
                                            'dec_deg_cont': 'dec'})

        self.sources = refcat


class Catalog:
    """Catalogue of high quality selavy components for comparative image quality analysis"""

    def __init__(self, image: Image, survey_name: str, **kwargs):

        self.image = image
        self.name = survey_name
        self.frequency = kwargs.get('frequency', self.image.frequency)
        self.isolationlim = kwargs.get('isolationlim', 45)
        self.snrlim = kwargs.get('snrlim', 0)

        logger.debug(f"Creating {self.name.upper()} Catalogue")
        self.sources = self.image.get_catalogue(self.name.upper())
        logger.debug(f"{len(self.sources)} {self.name.upper()} sources in field.")

        self._filter_isolated()
        self._filter_snr()
        self._filter_extended()

        logger.debug(f"{len(self.sources)} {self.name.upper()} sources after filtering.")

        if len(self.sources) == 0:
            msg = f"No remaining {self.name.upper()} sources in field {self.image.name}."
            raise AssertionError(msg)

        self._create_coords()
        self._add_distance_from_pos(self.image.fieldcentre)

    def __repr__(self):
        name = "<{}({}): {} sources>"
        return name.format(__class__.__name__, self.name, len(self.sources))

    def _add_distance_from_pos(self, position, label="centre"):
        self.sources["dist_{}".format(label)] = position.separation(self.coords).deg

    def _add_name_col(self):
        self.sources['name'] = self.coords.to_string('hmsdms').str.replace(' ', '_')

    def _create_coords(self):
        self.coords = SkyCoord(ra=self.sources.ra, dec=self.sources.dec, unit=u.deg)

    def _filter_extended(self):
        if self.name not in ['racs', 'ref']:
            return

        initial = len(self.sources)

        # Compactness (Hale et al. 2021)
        #      St/Sp < 1.025 + 0.69 * SNR^(-0.62)
        # --> (St/Sp - 1.025) * SNR^0.62 < 0.69
        SNR = self.sources.flux_int / self.sources.rms_image
        int_peak_ratio = self.sources.flux_int / self.sources.flux_peak

        self.sources = self.sources[
            ((int_peak_ratio - 1.025) * SNR**0.62 < 0.69)
        ].reset_index(drop=True)

        if self.name == 'racs':
            self.sources = self.sources[
                (self.sources.N_Gaus == 1)
            ].reset_index(drop=True)

        self.n_extended = initial - len(self.sources)
        logger.debug(f'{self.n_extended} {self.name.upper()} sources removed with extended filter')

    def _filter_isolated(self):

        initial = len(self.sources)
        if initial > 1:
            self._get_dist_to_neighbour()
            self.sources = self.sources[
                self.sources.d2neighbour > self.isolationlim * u.arcsec
            ].reset_index(drop=True)

        self.crowded = initial - len(self.sources)
        logger.debug(f'{self.crowded} {self.name.upper()} sources removed with isolation filter')

    def _filter_snr(self):
        if self.name == 'icrf':
            return

        initial = len(self.sources)

        self.sources = self.sources[
            self.sources.flux_int / self.sources.rms_image > self.snrlim].reset_index(drop=True)

        self.lowsnr = initial - len(self.sources)
        logger.debug(f'{self.lowsnr} {self.name.upper()} sources removed with SNR filter')

    def _get_dist_to_neighbour(self):
        coords = SkyCoord(ra=self.sources.ra, dec=self.sources.dec, unit=u.deg)
        idx, d2d, d3d = coords.match_to_catalog_sky(coords, nthneighbor=2)

        self.sources['d2neighbour'] = d2d.arcsec
