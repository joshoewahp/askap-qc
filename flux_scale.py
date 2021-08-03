import os
import click
import glob
import logging
import itertools as it
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from askap import Region
from logger import setupLogger

COLORS = ['darkgreen', 'cornflowerblue', 'mediumvioletred', 'mediumturquoise', 'darkorange', 'fuchsia',
          'mediumseagreen', 'rebeccapurple', 'teal', 'darkolivegreen', 'palegreen', 'gold']
MARKERS = ['d'] * len(COLORS)

logger = logging.getLogger(__name__)


@click.command()
@click.option('-f', '--fluxtype', type=click.Choice(['peak', 'int']), default='{fluxtype}',
              help='Option to compare {fluxtype} or integrated fluxes.')
@click.option('-b', '--bins', default=150,
              help='Number of bins in offset histograms.')
@click.option('-m', '--matchdir', default='matches/VASTP1/', type=click.Path(),
              help='Path to parent crossmatch directory')
@click.option('-r', '--rms', default=None, type=float,
              help='Plot flux scale variance with this typical RMS noise')
@click.option('-R', '--regions', multiple=True, default=None,
              help='Region numbers to include (e.g. -r 3 4)')
@click.option('-s', '--survey', default='sumss', type=click.Choice(['ref', 'racs', 'sumss', 'nvss']),
              help='Name of survey for astrometry comparison')
@click.option('-S', '--snrlim', default=10,
              help='Lower bound on source SNR.')
@click.option('-y', '--ylim', default=2,
              help='Upper bound on plotted flux ratio.')
@click.option('-v', '--verbose', is_flag=True, default=False,
              help="Enable verbose logging.")
def main(fluxtype, bins, matchdir, rms, regions, survey, snrlim, ylim, verbose):

    setupLogger(verbose)

    if regions:
        regions = [r for r in regions]
        fields = Region(regions).fields

    name = matchdir.split('/')[1]

    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(5, 6, figure=fig)
    ax1 = fig.add_subplot(gs[:, 0:5]) # offsets
    ax2 = fig.add_subplot(gs[:, 5:6]) # histogram

    ax1.set_xlabel(f'SB9602 Flux Density (mJy)')
    ax1.set_ylabel(f'GW / SB9602 Flux Ratio')
    ax2.set_xlabel('Count')

    all_epochs = []
    epochs = [e for e in os.listdir(matchdir) if '00' not in e]
    for color, epoch in zip(COLORS, epochs):

        s = 'askap' if survey in ['racs', 'ref'] else survey
        files = glob.glob(f'{matchdir}/{epoch}/*{s}.csv')
        if regions:
            files = [f for field in fields for f in files if field in f]
        
        fluxes = pd.concat([pd.read_csv(f) for f in files])
        fluxes = fluxes[fluxes[f'askap_flux_{fluxtype}'] / fluxes.askap_rms_image > snrlim]
        fluxes.insert(0, 'epoch', epoch)
        all_epochs.append(fluxes)

        epochmedian = fluxes[f'flux_{fluxtype}_ratio'].median()
        ax1.axhline(epochmedian,
                    ls=':',
                    color=color,
                    zorder=10,
                    label=f'{epoch} Median: {epochmedian:.2f}')

    all_epochs = pd.concat(all_epochs)
    all_epochs['flux_{fluxtype}_ratio'] = 1 / all_epochs[f'flux_{fluxtype}_ratio']

    med_maj = all_epochs.askap_maj_axis.median()
    med_min = all_epochs.askap_min_axis.median()
    bmaj = 15
    bmin = 12
    med_ratio = all_epochs[f'flux_{fluxtype}_ratio'].median()
    std_ratio = all_epochs[f'flux_{fluxtype}_ratio'].std()

    unique = all_epochs.drop_duplicates(subset=[f'{survey}_ra', f'{survey}_dec'])
    logger.info(f"{len(unique)} unique sources")
    logger.info(f'{len(all_epochs)} sources used')

    logger.info(f"Median flux density ratio of {med_ratio:.2f} +- {std_ratio:.2f}")

    all_epochs = all_epochs[all_epochs[f'flux_{fluxtype}_ratio'] < ylim]

    ax1.scatter(all_epochs[f'{s}_flux_{fluxtype}'],
                all_epochs[f'flux_{fluxtype}_ratio'],
                color='k', s=2, alpha=0.1, zorder=10)
    ax2.hist(all_epochs[f'flux_{fluxtype}_ratio'], histtype='step', color='k', bins=bins,
             orientation='horizontal')

    if rms:
        xaxis = np.linspace(all_epochs[f'askap_flux_{fluxtype}'].min(),
                            all_epochs[f'askap_flux_{fluxtype}'].max(), 100000)
        snr_unc = xaxis / med_ratio * np.sqrt(
            (std_ratio/med_ratio)**2 + (np.sqrt(2 * xaxis ** 2 / (med_maj * med_min * (xaxis/rms)**2 / (4 * bmaj * bmin)))/xaxis**2))
        yaxis = (xaxis + snr_unc) / xaxis
        yaxis2 = (xaxis - snr_unc) / xaxis

        ax1.plot(xaxis, yaxis, color='r', lw=2, ls='--')
        ax1.plot(xaxis, yaxis2, color='r', lw=2, ls='--')

    ax1.axhline(1, ls='-', lw=1, color='gray')
    ax2.axhline(1, ls='-', lw=1, color='gray')

    ax2.set_yticklabels([])
    ax2.set_yticks([])

    ax1.set_xscale('log')
    # ax1.set_yscale('log')
    # ax2.set_yscale('log')
    ax1.legend()

    # ax1.set_xlim([1, 100])
    ax1.set_ylim([0.2, 2])
    ax2.set_ylim([0.2, 2])

    fig.savefig('vastp1_reg3-4_fluxscale.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
