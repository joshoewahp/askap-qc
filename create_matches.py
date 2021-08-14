import click
import os
import time
import logging
from askap import Epoch, Region, Filepair
from logger import setupLogger
from matching import match_cats
from pathlib import Path
from catalog import ReferenceCatalog

logger = logging.getLogger(__name__)


@click.command()
@click.option('-d', '--dataset', type=click.Path(),
              help='Run on all images/selavy and epochs at this Path.')
@click.option('-e', '--epoch', type=click.Path(),
              help='Run on all images/selavy within epoch at this Path.')
@click.option('-f', '--field', type=click.Path(), nargs=2,
              help='Run on single field / selavy at these Paths.')
@click.option('-R', '--refcat', type=click.Path(), default=None,
              help='Location of reference catalogue table.')
@click.option('--combined/--no-combined', is_flag=True, default=True,
              help='Flag to use COMBINED mosaics, or otherwise raw TILES.')
@click.option('-S', '--stokes', type=click.Choice(['I', 'V']), default='I',
              help='Stokes parameter (either I or V).')
@click.option('-m', '--maxoffset', type=float, default=10,
              help='Maximum positional offset / association radius in arcsec.')
@click.option('-b', '--band', type=click.Choice(['low', 'mid']), default='low',
              help='Frequency band / footprint for region selections.')
@click.option('-r', '--region', multiple=True, default=None,
              help='Limit run to these Pilot survey region (band-dependent).')
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
         band, region, isolim, snrlim, savedir, verbose, wildcard):

    setupLogger(verbose, filename='qc.log')

    tiletype = 'COMBINED' if combined else 'TILES'

    region = Region(region, band=band)

    # Create Epoch objects
    if dataset and os.path.exists(dataset):
        if region:
            logger.warning("Region selections currently limited to one band")

        epoch_paths = sorted(Path(dataset).glob(wildcard))
        epochs = [Epoch(epoch, region, tiletype, stokes, band) for epoch in epoch_paths]
    elif epoch and os.path.exists(epoch):
        epochs = [Epoch(Path(epoch), region, tiletype, stokes, band)]
    elif field:
        epochs = None
    else:
        raise SystemExit("Must pass valid directory to either --dataset (-d), --epoch (-e), or --field (-f).")

    # Get reference catalogue. Use system hostname to determine
    refcat = ReferenceCatalog(refcat)

    # Match catalogues for all fields / epochs specified
    if epochs:

        for epoch in epochs:

            outdir = f'matches/{savedir}/{epoch.name}'
            os.makedirs(outdir, exist_ok=True)
            logger.info(f"Processing {epoch.num_files} images in {epoch.name}")
            logger.info(f'Saving output to {outdir}')

            for files in epoch.files:
                match_cats(files, refcat, maxoffset, isolim, snrlim, outdir)

    # or run on a single field
    else:

        image, sel = field
        files = Filepair(Path(image), Path(sel))

        outdir = f'matches/{savedir}/'
        os.makedirs(outdir, exist_ok=True)
        logger.info(f'Saving output to {outdir}')

        match_cats(files, refcat, maxoffset, isolim, snrlim, outdir)


if __name__ == '__main__':
    t1 = time.time()

    try:
        main()
    except Exception as e:
        logger.exception(e)
        exit(1)

    t2 = time.time()
    logger.info(f'Took {t2-t1:.1f} seconds')
