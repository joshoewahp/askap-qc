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
def main(bins, matchdir, rms, regions, survey, snrlim, ylim, verbose):

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
        fluxes = fluxes[fluxes.askap_flux_peak / fluxes.askap_rms_image > snrlim]
        fluxes.insert(0, 'epoch', epoch)
        all_epochs.append(fluxes)

        ax1.axhline(fluxes.flux_peak_ratio.median(), ls=':',
                    color=color, zorder=10,
                    label=f'{epoch} Median: {fluxes.flux_peak_ratio.median():.2f}')

    all_epochs = pd.concat(all_epochs)
    all_epochs['flux_peak_ratio'] = 1 / all_epochs.flux_peak_ratio

    med_maj = all_epochs.askap_maj_axis.median()
    med_min = all_epochs.askap_min_axis.median()
    bmaj = 15
    bmin = 12
    med_ratio = all_epochs.flux_peak_ratio.median()
    std_ratio = all_epochs.flux_peak_ratio.std()
   
    print(all_epochs)
    all_epochs.to_csv(f'vastp1_reg3-4_{s}_fluxes.csv')

    unique = all_epochs.drop_duplicates(subset=[f'{survey}_ra', f'{survey}_dec'])
    logger.info(f"{len(unique)} unique sources" )
    logger.info(f'{len(all_epochs)} sources used')

    all_epochs = all_epochs[all_epochs.flux_peak_ratio < ylim]

    ax1.scatter(all_epochs[f'{s}_flux_peak'],
                all_epochs.flux_peak_ratio, 
                color='k', s=2, alpha=0.4, zorder=10)
    ax2.hist(all_epochs.flux_peak_ratio, histtype='step', color='k', bins=bins,
             orientation='horizontal')

    if rms:
        xaxis = np.linspace(all_epochs.askap_flux_peak.min(), all_epochs.askap_flux_peak.max(), 100000)
        snr_unc = xaxis / med_ratio * np.sqrt(
            (std_ratio/med_ratio)**2 + (np.sqrt(2 * xaxis ** 2 / (med_maj * med_min * (xaxis/rms)**2 / (4 * bmaj * bmin)))/xaxis**2))
        yaxis = (xaxis + snr_unc) / xaxis
        yaxis2 = (xaxis - snr_unc) / xaxis

        ax1.plot(xaxis, yaxis, color='r', lw=2, ls='--')
        ax1.plot(xaxis, yaxis2, color='r', lw=2, ls='--')


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
