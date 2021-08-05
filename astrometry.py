import os
import click
import glob
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from askap import Region
from logger import setupLogger

COLORS = ['darkgreen', 'cornflowerblue', 'mediumvioletred', 'mediumturquoise', 'darkorange', 'fuchsia',
          'mediumseagreen', 'rebeccapurple', 'teal', 'darkolivegreen', 'palegreen', 'gold']
MARKERS = ['d'] * len(COLORS)

logger = logging.getLogger(__name__)

@click.command()
@click.option('-a', '--axlim', default=4, type=int,
              help='Axis range in arcsec.')
@click.option('-A', '--alpha', default=None, type=float,
              help='Alpha for scatter points.')
@click.option('-b', '--bins', default=None,
              help='Number of bins in offset histograms.')
@click.option('-S', '--snrlim', default=10,
              help='Lower bound on source SNR.')
@click.option('-R', '--regions', multiple=True, default=None,
              help='Region numbers to include (e.g. -r 3 4)')
@click.option('-s', '--survey', default='icrf', type=click.Choice(['icrf', 'ref', 'racs', 'sumss', 'nvss']),
              help='Name of survey for astrometry comparison')
@click.option('-v', "--verbose", is_flag=True, default=False,
              help="Enable verbose logging.")
@click.argument('matchdir', type=click.Path())
def main(axlim, alpha, bins, matchdir, snrlim, regions, survey, verbose):

    setupLogger(verbose)

    if regions:
        regions = [r for r in regions]
        fields = Region(regions).fields

    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(4, 4, figure=fig)
    ax1 = fig.add_subplot(gs[1:, :3]) # offsets
    ax2 = fig.add_subplot(gs[0, :3]) # ra hist
    ax3 = fig.add_subplot(gs[1:, 3]) # dec hist

    if not bins:
        bins = 250 if survey == 'racs' else 30
    if not alpha:
        alpha = 0.03 if survey == 'racs' else 0.3

    ax1.set_xlabel('RA Offset (arcsec)')
    ax1.set_ylabel('Dec Offset (arcsec)')
    ax1.set_xlim([-axlim, axlim])
    ax1.set_ylim([-axlim, axlim])

    ax2.set_ylabel('Count')
    ax3.set_xlabel('Count')

    # Plot pixel box, unpack line from returned tuple for legend
    pixbox, = ax1.plot([-1.25, -1.25, 1.25, 1.25, -1.25],
                       [-1.25, 1.25, 1.25, -1.25, -1.25],
                       ls='--',
                       lw=2,
                       alpha=1,
                       zorder=4,
                       color='gray')

    # Plot the gridlines
    for i in range(-axlim, axlim+1):
        ax1.axvline(i, color='k', alpha=0.1, zorder=1)
        ax1.axhline(i, color='k', alpha=0.1, zorder=1)
    
    all_epochs = []
    epochs = [e for e in os.listdir(matchdir) if '00' not in e]

    handles = []
    labels = []
    for color, marker, epoch in zip(COLORS, MARKERS, epochs):

        files = glob.glob(f'{matchdir}/{epoch}/*{survey}-askap.csv')
        if regions:
            files = [f for field in fields for f in files if field in f]

        if len(files) == 0:
            continue

        offsets = pd.concat([pd.read_csv(f) for f in files])
        offsets = offsets[offsets.askap_flux_peak/offsets.askap_rms_image > snrlim]
        offsets.insert(0, 'epoch', epoch)
        
        all_epochs.append(offsets)

        sc = ax1.scatter(offsets.ra_offset.median(), offsets.dec_offset.median(),
                         marker=marker, s=100, color=color, zorder=10, label=epoch)
        ax1.scatter(offsets.ra_offset, offsets.dec_offset, s=5, alpha=alpha,
                    zorder=3, color='k')

        handles.append(sc)
        labels.append(epoch)

    all_epochs = pd.concat(all_epochs)

    med_ra = all_epochs.ra_offset.median()
    med_dec = all_epochs.dec_offset.median()
    stderr_ra = all_epochs.ra_offset.std() / np.sqrt(len(all_epochs.ra_offset))
    stderr_dec = all_epochs.ra_offset.std() / np.sqrt(len(all_epochs.dec_offset))
    std_ra = all_epochs.ra_offset.std()
    std_dec = all_epochs.dec_offset.std()
    
    ax2.hist(all_epochs.ra_offset, bins=bins, histtype='step', color='k')
    ax3.hist(all_epochs.dec_offset, bins=bins, histtype='step', color='k',
             orientation='horizontal')

    medline = ax1.axvline(med_ra, ls='-', zorder=5, color='r')
    ax1.axvline(med_ra, ls='-', zorder=5, color='r')
    ax1.axhline(med_dec, ls='-', zorder=5, color='r')
    ax2.axvline(med_ra, ls='-', zorder=5, color='r')
    ax3.axhline(med_dec, ls='-', zorder=5, color='r')

    stdline = ax1.axvline(med_ra + std_ra, ls=':', alpha=0.5, zorder=5, color='r')
    ax1.axhline(med_dec + std_dec, ls=':', alpha=0.5, zorder=5, color='r')
    ax2.axvline(med_ra + std_ra, ls=':', alpha=0.5, zorder=5, color='r')
    ax3.axhline(med_dec + std_dec, ls=':', alpha=0.5, zorder=5, color='r')
    ax1.axvline(med_ra - std_ra, ls=':', alpha=0.5, zorder=5, color='r')
    ax1.axhline(med_dec - std_dec, ls=':', alpha=0.5, zorder=5, color='r')
    ax2.axvline(med_ra - std_ra, ls=':', alpha=0.5, zorder=5, color='r')
    ax3.axhline(med_dec - std_dec, ls=':', alpha=0.5, zorder=5, color='r')
    
    logger.info(f'Matching to {len(all_epochs)} {survey.upper()} measurements with SNR > {snrlim}')
    unique = all_epochs.drop_duplicates(subset=[f'{survey}_ra', f'{survey}_dec'])

    logger.info(f"{len(unique)} unique sources" )
    logger.info(f"Right ascension offset of {med_ra:.2f} +/- {std_ra:.2f} arcsec")
    logger.info(f"and standard error of {stderr_ra:.4f} arcsec")
    logger.info(f"Declination offset of {med_dec:.2f} +/- {std_dec:.2f} arcsec")
    logger.info(f"    and standard error of {stderr_dec:.4f} arcsec")

    ax1.set_xlim([-(axlim + .2), axlim + .2])
    ax1.set_ylim([-(axlim + .2), axlim + .2])
    ax2.set_xlim([-(axlim + .2), axlim + .2])
    ax3.set_ylim([-(axlim + .2), axlim + .2])

    ax2.set_xticks(range(-axlim, axlim+1))
    ax3.set_yticks(range(-axlim, axlim+1))
    ax1.set_xticks(range(-axlim, axlim+1))
    ax1.set_yticks(range(-axlim, axlim+1))
    
    leg1 = ax1.legend(handles=handles, labels=labels, loc=3, ncol=4)
    leg2 = ax1.legend(handles=[medline, stdline, pixbox],
                      labels=['median', 'stddev', 'single pixel'],
                      loc=1)
    ax1.add_artist(leg1)
    ax1.add_artist(leg2)
    
    # fig.savefig('vastp1_astrometry.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
