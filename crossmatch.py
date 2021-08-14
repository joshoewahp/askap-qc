#!/usr/env/bin python

import re
import logging
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from collections import defaultdict
from catalog import Catalog

logger = logging.getLogger(__name__)


class Crossmatch:
    """Crossmatch between two component catalogues"""

    def __init__(self, base_catalog: Catalog, comp_catalog: Catalog, **kwargs):

        self.base_cat = base_catalog
        self.comp_cat = comp_catalog
        self.base_freq = self.base_cat.frequency
        self.comp_freq = self.comp_cat.frequency
        self.spectral_index = kwargs.get('spectral_index', -.8)
        self.maxoffset = kwargs.get('maxoffset', 10) * u.arcsec
        self.performed = False

        self._perform_crossmatch()
        self.ra_offset_med = self.matches.ra_offset.median()
        self.dec_offset_med = self.matches.dec_offset.median()
        self.ra_offset_rms = np.sqrt(np.mean(np.square(self.matches.ra_offset)))
        self.dec_offset_rms = np.sqrt(np.mean(np.square(self.matches.dec_offset)))

        self._get_flux_ratios()

    def __repr__(self):
        return f"<{__class__.__name__}: {self.base_cat.survey_name}-{self.comp_cat.survey_name}>"

    def _perform_crossmatch(self):
        idx, d2d, _ = self.base_cat.coords.match_to_catalog_sky(self.comp_cat.coords)
        matches = self.comp_cat.sources.iloc[idx].reset_index(drop=True)
        self.matches = self.base_cat.sources.copy(deep=True)

        base_cols = {c: '{}_{}'.format(self.base_cat.survey_name, c) for c in self.matches.columns}
        comp_cols = {c: '{}_{}'.format(self.comp_cat.survey_name, c) for c in matches.columns}
        matches.rename(columns=comp_cols, inplace=True)
        self.matches.rename(columns=base_cols, inplace=True)
        self.matches = self.matches.join(matches)
        self.matches["d2d"] = d2d.arcsec

        self.matches = self.matches[self.matches.d2d <= self.maxoffset].reset_index(drop=True)
        assert len(self.matches) > 0, "No sufficient crossmatches in this field."
        self._get_offsets()
        self.matches.insert(0, 'image', self.base_cat.image.fieldname)
        self.performed = True

        logger.debug("Crossmatch complete.")

    def _get_offsets(self) -> tuple[np.array, np.array]:
        """Calculate and assign RA and Dec offsets between base_cat and comp_cat."""
        
        self.basecoords = SkyCoord(ra=self.matches['{}_ra'.format(self.base_cat.survey_name)],
                                    dec=self.matches['{}_dec'.format(self.base_cat.survey_name)],
                                    unit=u.deg)
        self.compcoords = SkyCoord(ra=self.matches['{}_ra'.format(self.comp_cat.survey_name)],
                                    dec=self.matches['{}_dec'.format(self.comp_cat.survey_name)],
                                    unit=u.deg)
        ra_offset, dec_offset = self.basecoords.spherical_offsets_to(self.compcoords)
        self.matches['ra_offset'] = ra_offset.arcsec
        self.matches['dec_offset'] = dec_offset.arcsec

        return ra_offset, dec_offset
        
    def _get_flux_ratios(self):

        # ICRF has no flux density measurements, and is used only for astrometry
        if self.comp_cat.survey_name == 'icrf':
            self.matches['flux_int_ratio'] = np.nan
            self.matches['flux_peak_ratio'] = np.nan

            return

        _freq_ratio = (self.base_freq / self.comp_freq)
        for fluxtype in ['flux_peak', 'flux_int']:
            base_flux = self.matches['{}_{}'.format(self.base_cat.survey_name, fluxtype)]
            comp_flux = self.matches['{}_{}'.format(self.comp_cat.survey_name, fluxtype)]
            new_flux_col = 'scaled_{}_{}'.format(self.comp_cat.survey_name, fluxtype)
            self.matches[new_flux_col] = _freq_ratio**self.spectral_index * comp_flux
            self.matches[f'{fluxtype}_ratio'] = base_flux / self.matches[new_flux_col]

        return 
    
    def save(self, outdir: str):
        fieldname = self.comp_cat.image.fieldname
        self.matches.to_csv(f'{outdir}/{fieldname}_{self.comp_cat.survey_name}-{self.base_cat.survey_name}.csv', index=False)
