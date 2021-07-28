import os
import click
import glob
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
from askap import Region
from matplotlib.gridspec import GridSpec
from logger import setupLogger

logger = logging.getLogger(__name__)

@click.command()
@click.option('-d', '--dataset', type=click.Path(),
              help='Run for all images and epochs at this Path.')
@click.option('-r', '--regions', multiple=True, default=None,
              help='Region numbers to include (e.g. -r 3 4)')
@click.option('-v', "--verbose", is_flag=True, default=False,
              help="Enable verbose logging.")
def main(dataset, regions, verbose):

    setupLogger(verbose)

    if regions:
        regions = [r for r in regions]
    else:
        regions = ['1', '2', '3', '4', '5', '6']

    fields = Region(regions).fields

    if not os.path.exists('matches/field_times.csv'):

        df = pd.DataFrame()
        epochs = []
        for epoch in sorted(os.listdir(dataset)):

            if epoch == 'EPOCH00' or 'EPOCH' not in epoch:
                continue

            epochs.append(epoch)

            imagepath = f"{epoch}/COMBINED/STOKESI_IMAGES/*.fits"
            files = sorted(glob.glob(dataset + imagepath))
            images = [f for field in fields for f in files if field in f]

            times = dict()
            for i, image in enumerate(images):
                name = image.split('/')[-1].split('.')[0]
                hdul = fits.open(image)

                header = hdul[0].header
                times[name] = Time(header.get('MJD-OBS'), format='mjd').datetime

            epochdf = pd.DataFrame.from_dict({epoch: times})
            df = pd.concat([df, epochdf], axis=1)

        df = df.reset_index().rename(columns={'index': 'Field'})
            
        df.to_csv('matches/field_times.csv', index=False)
    
    df = pd.read_csv('matches/field_times.csv')

    logger.info(f'\n{df}')
   

if __name__ == '__main__':
    main()
