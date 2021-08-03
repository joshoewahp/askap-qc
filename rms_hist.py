import os
import click
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.gridspec import GridSpec
from askap import Region


@click.command()
@click.option('-d', '--dataset', type=click.Path(),
              help='Run for all images and epochs at this Path.')
@click.option("-S", "--stokes", type=click.Choice(['I', 'V']), default='I',
              help="Stokes parameter (either I or V).")
@click.option('-r', '--regions', multiple=True, default=None,
              help='Region numbers to include (e.g. -r 3 4)')
@click.option("--combined/--no-combined", is_flag=True, default=True,
              help="Flag to use COMBINED mosaics, or otherwise raw TILES.")
@click.option('-v', "--verbose", is_flag=True, default=False,
              help="Enable verbose logging.")
def main(dataset, stokes, regions, combined, verbose):


    if regions:
        regions = [r for r in regions]
    else:
        regions = ['1', '2', '3', '4', '5', '6']

    fields = Region(regions).fields

    if not os.path.exists('matches/field_rms.csv'):
        print('creating')
        tiletype = 'COMBINED' if combined else 'TILES'

        df = pd.DataFrame()
        epochs = []
        for epoch in sorted(os.listdir(dataset)):

            if epoch == 'EPOCH00' or 'EPOCH' not in epoch:
                continue

            epochs.append(epoch)

            imagepath = f"{epoch}/{tiletype}/STOKES{stokes}_RMSMAPS/*rms.fits"
            files = sorted(glob.glob(dataset + imagepath))
            images = [f for field in fields for f in files if field in f]

            rmsvals = dict()
            for i, image in enumerate(images):
                name = image.split('/')[-1].split('.')[0]
                hdul = fits.open(image)

                data = hdul[0].data
                data = data[np.where(~np.isnan(data))]
                meanrms = np.median(data) * 1000
                rmsvals[name] = meanrms

            epochdf = pd.DataFrame.from_dict({epoch: rmsvals})
            df = pd.concat([df, epochdf], axis=1)


        df.to_csv('matches/field_rms.csv')
    
    df = pd.read_csv('matches/field_rms.csv')
    df.set_index('Unnamed: 0', inplace=True)
    # bins = 75
    # axlim = 4

    print(df)
    fig = plt.figure(figsize=(4, 4))
    ax1 = fig.add_subplot() 

    rms = [r for c in df.columns for r in df[c].values if not np.isnan(r)]
    print(np.median(rms))

    ax1.hist(rms, bins=20, histtype='step', color='k')
    
    ax1.set_xlabel('Median Image RMS Noise (mJy)')
    ax1.set_ylabel('Count')
    plt.show()
    fig.savefig("vastp1_reg3-4_rmshist.png", bbox_inches='tight')
    
    # gs = GridSpec(4, 4, figure=fig)
    # ax1 = fig.add_subplot(gs[1:, :3]) # offsets
    # ax2 = fig.add_subplot(gs[0, :3]) # ra hist
    # ax3 = fig.add_subplot(gs[1:, 3]) # dec hist

    # ax1.set_xlabel('RA Offset (arcsec)')
    # ax1.set_ylabel('Dec Offset (arcsec)')
    # ax1.set_xlim([-axlim, axlim])
    # ax1.set_ylim([-axlim, axlim])

#     all_epochs = []
#     epochs = [e for e in os.listdir('matches') if not any([s in e for s in ['00', 'x']])]
#     for color, epoch in zip(COLORS, epochs):

#         files = glob.glob(f'matches/{epoch}/*icrf.csv')
#         offsets = pd.concat([pd.read_csv(f) for f in files])
#         all_epochs.append(offsets)

#     all_epochs = pd.concat(all_epochs)

#     ax1.legend()
    
#     plt.show()

if __name__ == '__main__':
    main()
