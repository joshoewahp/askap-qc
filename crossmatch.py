#!/usr/local/bin/python3

import os
import re
import logging
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from collections import defaultdict


class Crossmatch:
    """
    docstring for crossmatch
    """

    def __init__(self, base_catalog, comp_catalog, **kwargs):
        self.base_cat = base_catalog
        self.comp_cat = comp_catalog
        self.base_freq = self.base_cat.frequency
        self.comp_freq = self.comp_cat.frequency
        self.spectral_index = kwargs.get('spectral_index', -.8)
        self.performed = False
        self.maxsep = kwargs.get('maxsep', 10) * u.arcsec

        self.logger = logging.getLogger(__name__)

        self._perform_crossmatch()
        self.ra_offset_med = self.df.ra_offset.median()
        self.dec_offset_med = self.df.dec_offset.median()
        self.ra_offset_rms = np.sqrt(np.mean(np.square(self.df.ra_offset)))
        self.dec_offset_rms = np.sqrt(np.mean(np.square(self.df.dec_offset)))

        if kwargs.get('scale_flux', True):
            self._get_flux_ratios()
        else:
            self.df['flux_int_ratio'] = np.nan
            self.df['flux_peak_ratio'] = np.nan

    def __repr__(self):
        name = "{}({}-{}): {} matches"
        return name.format(__class__.__name__,
                           self.base_cat.name,
                           self.comp_cat.name,
                           self.df.shape[0])

    def _perform_crossmatch(self):
        idx, d2d, d3d = self.base_cat.coords.match_to_catalog_sky(self.comp_cat.coords)
        matches = self.comp_cat.sources.iloc[idx].reset_index(drop=True)
        self.df = self.base_cat.sources.copy(deep=True)

        base_cols = {c: '{}_{}'.format(self.base_cat.name, c) for c in self.df.columns}
        comp_cols = {c: '{}_{}'.format(self.comp_cat.name, c) for c in matches.columns}
        matches.rename(columns=comp_cols, inplace=True)
        self.df.rename(columns=base_cols, inplace=True)
        self.df = self.df.join(matches)
        self.df["d2d"] = d2d.arcsec

        self.df = self.df[self.df.d2d <= self.maxsep].reset_index(drop=True)
        assert len(self.df) > 0, "No sufficient crossmatches in this field."
        self._get_offsets()
        self.df.insert(0, 'image', self.base_cat.image.name)
        self.performed = True
        self.logger.debug("Crossmatch complete.")

    def extend_crossmatch(self, catalog, prefix='vastp'):
        
        avg_ra, avg_dec = self._get_avg_coords(prefix)
        avg_coord = SkyCoord(ra=avg_ra, dec=avg_dec, unit=u.deg)
        cat_coord = SkyCoord(ra=catalog.sources.ra, dec=catalog.sources.dec, unit=u.deg)
        idx, d2d, d3d = avg_coord.match_to_catalog_sky(cat_coord)
        matches = catalog.sources.reset_index(drop=True)
        comp_cols = {c: '{}_{}'.format(catalog.name, c) for c in matches.columns}
        matches.rename(columns=comp_cols, inplace=True)
        self.df['idx'] = idx
        self.df['d2d'] = d2d.arcsec
        self.df = self.df.merge(matches, left_on='idx', right_index=True)

        col_select = list(comp_cols.values()) + ['d2d']
        self.df.loc[self.df.d2d > self.maxsep, col_select] = np.nan

        self.df.drop(columns=['d2d', 'idx'], inplace=True)
        assert len(self.df) > 0, "No sufficient crossmatches in this field."

    def _get_avg_coords(self, prefix):
        ra_cols, dec_cols = self._get_coord_cols(prefix)

        tempdf = self.df.copy()[ra_cols + dec_cols]
        
        # Handle RA wrapping
        ra_wrap_mask = tempdf[ra_cols] <= 0.5
        tempdf[ra_wrap_mask] += 360.
        tempdf['avg_ra'] = tempdf[ra_cols].mean(skipna=True, axis=1)
        tempdf['avg_dec'] = tempdf[dec_cols].mean(skipna=True, axis=1)

        # Undo wrapping offset
        ra_wrap_mask = tempdf.avg_ra >= 360.
        tempdf.loc[ra_wrap_mask, 'avg_ra'] -= 360.

        return tempdf.avg_ra, tempdf.avg_dec

    def _get_coord_cols(self, prefix):
        ra_pattern = re.compile(f'{prefix}\d*x*_ra')
        dec_pattern = re.compile(f'{prefix}\d*x*_dec')

        ra_cols = ['racs_ra'] + list(filter(ra_pattern.match, self.df.columns))
        dec_cols = ['racs_dec'] + list(filter(dec_pattern.match, self.df.columns))

        return ra_cols, dec_cols

    def _get_offsets(self, avg=False, epoch=None, prefix='vastp'):
        """
        Calculate and assign RA and Dec offsets between base_cat and comp_cat.
        """
        
        if not avg:
            self.basecoords = SkyCoord(ra=self.df['{}_ra'.format(self.base_cat.name)],
                                       dec=self.df['{}_dec'.format(self.base_cat.name)],
                                       unit=u.deg)
            self.compcoords = SkyCoord(ra=self.df['{}_ra'.format(self.comp_cat.name)],
                                       dec=self.df['{}_dec'.format(self.comp_cat.name)],
                                       unit=u.deg)
            ra_offset, dec_offset = self.compcoords.spherical_offsets_to(self.basecoords)
            self.df['ra_offset'] = ra_offset.arcsec
            self.df['dec_offset'] = dec_offset.arcsec

            return ra_offset, dec_offset

        else:
            ra_cols, dec_cols = self._get_coord_cols(prefix)
            if epoch is None:
                avg_ra, avg_dec = self._get_avg_coords(prefix)
                avg_coords = SkyCoord(ra=avg_ra, dec=avg_dec, unit=u.deg)
            else:
                try:
                    avg_coords = SkyCoord(ra=self.df[f'{epoch}_ra'],
                                          dec=self.df[f'{epoch}_dec'],
                                          unit=u.deg)
                except KeyError:
                    self.logger.warning(f"No {epoch} offsets for this field, defaulting to vastp1")
                    avg_coords = SkyCoord(ra=self.df['vastp1_ra'],
                                          dec=self.df['vastp1_dec'],
                                          unit=u.deg)

            for r, d in zip(ra_cols, dec_cols):
                comp_coords = SkyCoord(ra=self.df[r], dec=self.df[d], unit=u.deg)
                ra_off, dec_off = comp_coords.spherical_offsets_to(avg_coords)
                self.df[f'{r}_offset'] = ra_off.arcsec
                self.df[f'{d}_offset'] = dec_off.arcsec

            ra_offset = pd.concat([self.df[f'{col}_offset'] for col in ra_cols])
            dec_offset = pd.concat([self.df[f'{col}_offset'] for col in dec_cols])

            return ra_offset, dec_offset

    def _get_flux_ratios(self, avg=False, epoch=None, prefix='vastp'):

        if avg:
            int_pattern = re.compile(f'{prefix}\d*x*_flux_int')
            peak_pattern = re.compile(f'{prefix}\d*x*_flux_peak')
            int_cols = ['racs_flux_int'] + list(filter(st_pattern.match, self.df.columns))
            peak_cols = ['racs_flux_peak'] + list(filter(sp_pattern.match, self.df.columns))

            if epoch is None:
                med_flux_int = np.nanmedian(self.df[int_cols], axis=1)
                med_flux_peak = np.nanmedian(self.df[peak_cols], axis=1)
            else:
                try:
                    med_flux_int = self.df[f'{epoch}_flux_int']
                    med_flux_peak = self.df[f'{epoch}_flux_peak']
                except KeyError:
                    self.logger.warning(f"No {epoch} fluxes for this field, defaulting to vastp1.")
                    med_flux_int = self.df[f'vastp1_flux_int']
                    med_flux_peak = self.df[f'vastp1_flux_peak']

            self.med_flux = defaultdict(dict)

            for col in int_cols:
                self.df[f'{col}_ratio'] = self.df[col] / med_flux_int
                self.med_flux[col] = np.nanmedian(self.df[f'{col}_ratio'])
                self.med_flux[f'{col}_std'] = np.std(self.df[f'{col}_ratio'])
                # self.med_flux[f'{col}_std'] = np.sqrt(np.mean(np.square(self.df[f'{col}_ratio'])))
            for col in peak_cols:
                self.df[f'{col}_ratio'] = self.df[col] / med_flux_peak
                self.med_flux[col] = np.nanmedian(self.df[f'{col}_ratio'])
                self.med_flux[f'{col}_std'] = np.std(self.df[f'{col}_ratio'])
                # self.med_flux[f'{col}_rms'] = np.sqrt(np.mean(np.square(self.df[f'{col}_ratio'])))

            flux_int_ratio = pd.concat([self.df[f'{col}_ratio'] for col in int_cols])
            flux_peak_ratio = pd.concat([self.df[f'{col}_ratio'] for col in peak_cols])

            return flux_int_ratio, flux_peak_ratio

        else:
            
            _freq_ratio = (self.base_freq / self.comp_freq)
            for fluxtype in ['flux_peak', 'flux_int']:
                base_flux = self.df['{}_{}'.format(self.base_cat.name, fluxtype)]
                comp_flux = self.df['{}_{}'.format(self.comp_cat.name, fluxtype)]
                new_flux_col = 'scaled_{}_{}'.format(self.comp_cat.name, fluxtype)
                self.df[new_flux_col] = _freq_ratio**self.spectral_index * comp_flux
                self.df[f'{fluxtype}_ratio'] = base_flux / self.df[new_flux_col]

            return 
    
    def write_crossmatch(self, outname):
        if self.performed:
            self.df.to_csv(outname, index=False)
            self.logger.info("Written crossmatch dataframe to {}.".format(outname))
        else:
            self.logger.error("Need to perform a cross match first!")

    def calculate_ratio(self, col1, col2, output_col_name, col1_scaling=1., col2_scaling=1.,
                        dualmode=False, basecat="sumss"):
        self.df[output_col_name] = (self.df[col1] * col1_scaling) / (self.df[col2] * col2_scaling)

    def calculate_diff(self, col1, col2, output_col_name, col1_scaling=1.,
                       col2_scaling=1., dualmode=False, basecat="sumss"):
        self.df[output_col_name] = (self.df[col1] * col1_scaling) - (self.df[col2] * col2_scaling)
