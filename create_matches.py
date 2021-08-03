import click
import glob
import os
import time
import logging
import pandas as pd
from astropy.table import Table
from astropy.time import Time
from askap import Epoch, Filepair
from concurrent.futures import ProcessPoolExecutor, as_completed
from logger import setupLogger
from matching import match_cats
from pathlib import Path

from astropy.units import UnitsWarning
import warnings
warnings.filterwarnings('ignore', category=UnitsWarning, append=True)

logger = logging.getLogger(__name__)


@click.command()
@click.option('-d', '--dataset', type=click.Path(),
              help='Run on all images/selavy and epochs at this Path.')
@click.option('-e', '--epoch', type=click.Path(),
              help='Run on all images/selavy within epoch at this Path.')
@click.option('-f', '--field', type=click.Path(), nargs=2,
              help='Run on single field / selavy at these Paths.')
@click.option('-R', '--refcat', type=click.Path(),
              default='RACS-25asec-Mosaiced_Gaussians_Final_GalCut_v2021_03_01.fits',
              help='Location of reference catalogue table.')
@click.option('--combined/--no-combined', is_flag=True, default=True,
              help='Flag to use COMBINED mosaics, or otherwise raw TILES.')
@click.option('-S', '--stokes', type=click.Choice(['I', 'V']), default='I',
              help='Stokes parameter (either I or V).')
@click.option('-m', '--maxoffset', type=float, default=10,
              help='Maximum positional offset / association radius in arcsec.')
@click.option('-b', '--band', type=click.Choice(['low', 'mid']), default='low',
              help='Frequency band / footprint for region selections.')
@click.option('-r', '--regions', multiple=True, default=None,
              help='Limit run to these Pilot survey regions (band-dependent).')
@click.option('-i', '--isolim', type=float, default=150,
              help='Minimum nearest neighbour distance in arcsec.')
@click.option('-s', '--snrlim', type=float, default=10,
              help='Minimum component SNR across all surveys.')
@click.option('-S', '--savedir', type=str, default='results',
              help='Name of subdirectory to matches/ in which to save results')
@click.option('-v', '--verbose', is_flag=True, default=False,
              help='Enable verbose logging.')
@click.option('-w', '--wildcard', type=str, default='*EPOCH*',
              help='Regex to mark each epoch in a dataset.')
def main(dataset, epoch, field, refcat, combined, stokes, maxoffset,
         band, regions, isolim, snrlim, savedir, verbose, wildcard):

    setupLogger(verbose, filename='qc.log')
    
    tiletype = 'COMBINED' if combined else 'TILES'

    # Create Epoch objects
    if dataset and os.path.exists(dataset):
        if regions:
            logger.warning("Region selections currently limited to one band")

        epochs = sorted(Path(dataset).glob(wildcard))
        epochs = [Epoch(epoch, tiletype, stokes, regions, band) for epoch in epochs]
    elif epoch and os.path.exists(epoch):
        epochs = [Epoch(epoch, tiletype, stokes, regions, band)]
    elif field:
        epochs = None
    else:
        raise SystemExit("Must pass valid directory to either --dataset (-d), --epoch (-e), or --field (-f).")

    if 'RACS' in refcat:
        # Get RACS catalogue as base
        refcat = Table.read(refcat).to_pandas()
        columns = {'N_Gaus': 'N_Gaus',
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
                   'Noise': 'rms_image',}
        refcat = refcat.rename(columns=columns)[columns.values()]
    else:
        refcat = pd.read_fwf(refcat, skiprows=[1,])
        refcat = refcat.rename(columns={'ra_deg_cont': 'ra',
                                        'dec_deg_cont': 'dec'})
    
    if epochs:

        for epoch in epochs:
            os.makedirs(f'matches/{savedir}/{epoch.name}', exist_ok=True)

            logger.info(f"Processing {epoch.num_images} images in {epoch}")

            for files in epoch.files:
                catalogues = match_cats(files, refcat, maxoffset, isolim, snrlim)
                for k, v in catalogues.items():
                    v.df.to_csv(f'matches/{savedir}/{epoch.name}/{v.base_cat.image.name}_{k}.csv', index=False)


    else:
        # Run on single field
        files = Filepair(*field)
        name = files.image.split('/')[1]

        os.makedirs(f'matches/{savedir}/{name}', exist_ok=True)
        catalogues = match_cats(files, refcat, maxoffset, isolim, snrlim)
        for k, v in catalogues.items():
            v.df.to_csv(f'matches/{savedir}/{name}/{k}.csv', index=False)

      
if __name__ == '__main__':
    t1 = time.time()

    try:
        main()
    except Exception as e:
        logger.exception(e)
        
    t2 = time.time()
    logger.info(f'Took {t2-t1:.1f} seconds')
