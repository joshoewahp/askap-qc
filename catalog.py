import os
import logging
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from askap import Image


class Catalog:
    """
    docstring for survey
    """

    def __init__(self, image, survey_name, **kwargs):

        self.image = image
        self.name = survey_name
        self.frequency = kwargs.get('frequency', self.image.frequency)
        self.isolationlim = kwargs.get('isolationlim', 45)
        self.snrlim = kwargs.get('snrlim', 0)

        self.logger = logging.getLogger(__name__ + f' - {self.image.name}')

        self.logger.debug(f"Creating {self.name.upper()} Catalogue")

        self.sources = self.image.get_catalogue(self.name.upper())
        self.logger.debug(f"{len(self.sources)} {self.name.upper()} sources in field.")
            
        self._filter_isolated()
        self._filter_snr()
        self._filter_extended()
        
        self.logger.debug(f"{len(self.sources)} {self.name.upper()} sources after filtering.")
        assert len(self.sources) > 0, f"No remaining {self.name.upper()} sources in field {self.image.name}."

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
        int_peak_ratio = self.sources.flux_int/self.sources.flux_peak

        self.sources = self.sources[
            ((int_peak_ratio - 1.025) * SNR**0.62 < 0.69)
        ].reset_index(drop=True)

        if self.name == 'racs':
            self.sources = self.sources[
                (self.sources.N_Gaus == 1)
            ].reset_index(drop=True)

        self.n_extended = initial - len(self.sources)
        self.logger.debug(f'{self.n_extended} {self.name.upper()} sources removed with extended filter')

    def _filter_isolated(self):

        initial = len(self.sources)
        if initial > 1:
            self._get_dist_to_neighbour()
            self.sources = self.sources[self.sources.d2neighbour >
                                        self.isolationlim * u.arcsec].reset_index(drop=True)

        self.crowded = initial - len(self.sources)
        self.logger.debug(f'{self.crowded} {self.name.upper()} sources removed with isolation filter')

    def _filter_snr(self):
        if self.name == 'icrf':
            return

        initial = len(self.sources)

        self.sources = self.sources[
            self.sources.flux_int / self.sources.rms_image > self.snrlim].reset_index(drop=True)

        self.lowsnr = initial - len(self.sources)
        self.logger.debug(f'{self.lowsnr} {self.name.upper()} sources removed with SNR filter')

    def _get_dist_to_neighbour(self):
        coords = SkyCoord(ra=self.sources.ra, dec=self.sources.dec, unit=u.deg)
        idx, d2d, d3d = coords.match_to_catalog_sky(coords, nthneighbor=2)
        self.sources['d2neighbour'] = d2d.arcsec

    def _parse_coords(self):
        """
        Parses DataFrame columns for ra and dec data.
        (May in future just do forced cleaning of catalogs to consistent standard)

        Args:
           self: Catalog self instance

        Returns:
            None

        """
        self.sources['ra'] = self.sources.filter(regex='ra$|_RAJ2000').iloc[:, 0]
        self.sources['dec'] = self.sources.filter(regex='dec$|_DEJ2000').iloc[:, 0]

    def _sumss_rms(self, dec):

        if dec > -50:
            return 0.002
        else:
            return 0.0012

    def write_ann(self, name=None, color="GREEN"):

        if not name:
            name = self.name + ".ann"

        with open(name, 'w') as f:
            f.write("COORD W\n")
            f.write("PA STANDARD\n")
            f.write("COLOR {}\n".format(color))
            f.write("FONT hershey14\n")
            for i, row in self.sources.iterrows():
                f.write("ELLIPSE {} {} {} {} {}\n".format(
                    self.coords[i].ra.deg,
                    self.coords[i].dec.deg,
                    row.maj_axis / 3600,
                    row.min_axis / 3600,
                    row.pos_ang))
        self.logger.info("Wrote annotation file {}.".format(name))
