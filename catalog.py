import logging
import os
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table import Table
from dataclasses import dataclass

from askap import Image
from fileio import load_selavy_file
from pathlib import Path

logger = logging.getLogger(__name__)


def get_RACS_path(system):
    
    if system == 'vast-data':
        racs_path = '/home/joshoewahp/RACS-25asec-Mosaiced_Gaussians_Final_GalCut_v2021_03_01.fits'
    elif system == 'ada.physics.usyd.edu.au':
        racs_path = '/import/ada1/jpri6587/data/RACS-25asec-Mosaiced_Gaussians_Final_GalCut_v2021_03_01.fits'
    else:
        raise SystemExit("RACS / reference catalogue must be manually specified with --refcat (-R) unless on ada or nimbus.")

    return racs_path


class ReferenceCatalog:
    """Source catalogue as reference for flux and astrometry comparisons

    catpath should point either to the published RACS catalogue
    as an astropy table or a raw selavy component catalogue

    RACS catalogue will be automatically detected with catpath=None 
    if on ada or nimbus
    """

    def __init__(self, catpath: Path = None):

        system = os.uname()[1]
        if not catpath:
            catpath = get_RACS_path(system)

        self._set_reference_catalogue(catpath)

    def _set_reference_catalogue(self, catpath: Path) -> pd.DataFrame:

        if isinstance(catpath, str):
            catpath = Path(catpath)

        if 'RACS' in catpath.name:

            # Get RACS catalogue as base
            self.name = 'racs'

            refcat = Table.read(catpath).to_pandas()
            columns = {
                'N_Gaus': 'N_Gaus',
                'RA': 'ra',
                'Dec': 'dec',
                'E_RA': 'ra_err',
                'E_Dec': 'dec_err',
                'Total_flux_Component': 'flux_int',
                'E_Total_flux_Component': 'flux_int_err',
                'Peak_flux': 'flux_peak',
                'E_Peak_flux': 'flux_peak_err',
                'Maj': 'maj_axis',
                'Min': 'min_axis',
                'PA': 'pos_ang',
                'Separation_Tile_Centre': 'field_centre_dist',
                'Noise': 'rms_image'
            }
            refcat = refcat.rename(columns=columns)[columns.values()]
        else:

            self.name = 'ref'

            refcat = load_selavy_file(catpath)
            refcat['N_Gaus'] = np.nan
            refcat['field_centre_dist'] = np.nan
            refcat = refcat.rename(columns={'ra_deg_cont': 'ra',
                                            'dec_deg_cont': 'dec'})

        self.sources = refcat


@dataclass
class Catalog:
    """Catalogue of high quality selavy components for comparative image quality analysis"""

    image: Image
    survey_name: str
    frequency: float = None
    isolationlim: float = 150
    snrlim: float = 0
    
    def __post_init__(self):
        if not self.frequency:
            self.frequency = self.image.frequency
        self.sources = self.image.get_catalogue(self.survey_name.upper())

        logger.debug(f"{len(self.sources)} {self.survey_name.upper()} sources in field.")
        logger.debug(f"Creating {self.survey_name.upper()} Catalogue")

        self._filter_isolated()
        self._filter_snr()
        self._filter_extended()

        logger.debug(f"{len(self.sources)} {self.survey_name.upper()} sources after filtering.")

        if len(self.sources) == 0:
            msg = f"No remaining {self.survey_name.upper()} sources in field {self.image.fieldname}."
            raise AssertionError(msg)

        self._create_coords()
        self._add_distance_from_pos(self.image.fieldcentre)


    def _add_distance_from_pos(self, position, label="centre"):
        self.sources["dist_{}".format(label)] = position.separation(self.coords).deg

    def _create_coords(self):
        self.coords = SkyCoord(ra=self.sources.ra, dec=self.sources.dec, unit=u.deg)

    def _filter_extended(self):
        if self.survey_name not in ['racs', 'ref']:
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

        if self.survey_name == 'racs':
            self.sources = self.sources[
                (self.sources.N_Gaus == 1)
            ].reset_index(drop=True)

        self.n_extended = initial - len(self.sources)
        logger.debug(f'{self.n_extended} {self.survey_name.upper()} sources removed with extended filter')

    def _filter_isolated(self):

        initial = len(self.sources)
        if initial > 1:
            self._get_dist_to_neighbour()
            self.sources = self.sources[
                self.sources.d2neighbour > self.isolationlim * u.arcsec
            ].reset_index(drop=True)

        self.crowded = initial - len(self.sources)
        logger.debug(f'{self.crowded} {self.survey_name.upper()} sources removed with isolation filter')

    def _filter_snr(self):
        if self.survey_name == 'icrf':
            return

        initial = len(self.sources)

        self.sources = self.sources[
            self.sources.flux_int / self.sources.rms_image > self.snrlim].reset_index(drop=True)

        self.lowsnr = initial - len(self.sources)
        logger.debug(f'{self.lowsnr} {self.survey_name.upper()} sources removed with SNR filter')

    def _get_dist_to_neighbour(self):
        coords = SkyCoord(ra=self.sources.ra, dec=self.sources.dec, unit=u.deg)
        _, d2d, _ = coords.match_to_catalog_sky(coords, nthneighbor=2)

        self.sources['d2neighbour'] = d2d.arcsec
