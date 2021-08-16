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
@click.option('-S', '--save', type=click.Path(), default=None)
@click.option('-v', "--verbose", is_flag=True, default=False,
              help="Enable verbose logging.")
@click.argument('matchdir', type=click.Path())
def main(axlim, alpha, bins, matchdir, snrlim, regions, survey, save, verbose):

    setupLogger(verbose)

    if regions:
        regions = [r for r in regions]
        fields = Region(regions).fields

    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(4, 4, figure=fig)
    offset_ax = fig.add_subplot(gs[1:, :3])
    ra_ax = fig.add_subplot(gs[0, :3])
    dec_ax = fig.add_subplot(gs[1:, 3])

    if not bins:
        bins = 250 if survey == 'racs' else 30
    if not alpha:
        alpha = 0.03 if survey == 'racs' else 0.3

    offset_ax.set_xlabel('RA Offset (arcsec)')
    offset_ax.set_ylabel('Dec Offset (arcsec)')
    offset_ax.set_xlim([-axlim, axlim])
    offset_ax.set_ylim([-axlim, axlim])

    ra_ax.set_ylabel('Count')
    dec_ax.set_xlabel('Count')

    # Plot pixel box, unpack Line2D object from returned tuple for legend
    pixbox, = offset_ax.plot([-1.25, -1.25, 1.25, 1.25, -1.25],
                             [-1.25, 1.25, 1.25, -1.25, -1.25],
                             ls='--',
                             lw=2,
                             alpha=1,
                             zorder=4,
                             color='gray')

    # Plot the gridlines
    for i in range(-axlim, axlim+1):
        offset_ax.axvline(i, color='k', alpha=0.1, zorder=1)
        offset_ax.axhline(i, color='k', alpha=0.1, zorder=1)
    

    all_epochs = []
    epochs = [e for e in os.listdir(matchdir) if '00' not in e]

    handles = []
    labels = []
    for color, epoch in zip(COLORS, epochs):

        files = glob.glob(f'{matchdir}/{epoch}/*{survey}-askap.csv')
        if regions:
            files = [f for field in fields for f in files if field in f]

        if len(files) == 0:
            continue

        offsets = pd.concat([pd.read_csv(f) for f in files])
        offsets = offsets[offsets.askap_flux_peak/offsets.askap_rms_image > snrlim]
        offsets.insert(0, 'epoch', epoch)
        
        all_epochs.append(offsets)

        sc = offset_ax.scatter(offsets.ra_offset.median(), offsets.dec_offset.median(),
                         marker='d', s=100, color=color, zorder=10, label=epoch)
        offset_ax.scatter(offsets.ra_offset, offsets.dec_offset, s=5, alpha=alpha,
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
    
    ra_ax.hist(all_epochs.ra_offset, bins=bins, histtype='step', color='k')
    dec_ax.hist(all_epochs.dec_offset, bins=bins, histtype='step', color='k',
             orientation='horizontal')

    medline = offset_ax.axvline(med_ra, ls='-', zorder=5, color='r')
    offset_ax.axvline(med_ra, ls='-', zorder=5, color='r')
    offset_ax.axhline(med_dec, ls='-', zorder=5, color='r')
    ra_ax.axvline(med_ra, ls='-', zorder=5, color='r')
    dec_ax.axhline(med_dec, ls='-', zorder=5, color='r')

    stdline = offset_ax.axvline(med_ra + std_ra, ls=':', alpha=0.5, zorder=5, color='r')
    offset_ax.axhline(med_dec + std_dec, ls=':', alpha=0.5, zorder=5, color='r')
    ra_ax.axvline(med_ra + std_ra, ls=':', alpha=0.5, zorder=5, color='r')
    dec_ax.axhline(med_dec + std_dec, ls=':', alpha=0.5, zorder=5, color='r')
    offset_ax.axvline(med_ra - std_ra, ls=':', alpha=0.5, zorder=5, color='r')
    offset_ax.axhline(med_dec - std_dec, ls=':', alpha=0.5, zorder=5, color='r')
    ra_ax.axvline(med_ra - std_ra, ls=':', alpha=0.5, zorder=5, color='r')
    dec_ax.axhline(med_dec - std_dec, ls=':', alpha=0.5, zorder=5, color='r')
    
    logger.info(f'Matching to {len(all_epochs)} {survey.upper()} measurements with SNR > {snrlim}')
    unique = all_epochs.drop_duplicates(subset=[f'{survey}_ra', f'{survey}_dec'])

    logger.info(f"{len(unique)} unique sources" )
    logger.info(f"Right ascension offset of {med_ra:.2f} +/- {std_ra:.2f} arcsec")
    logger.info(f"and standard error of {stderr_ra:.4f} arcsec")
    logger.info(f"Declination offset of {med_dec:.2f} +/- {std_dec:.2f} arcsec")
    logger.info(f"and standard error of {stderr_dec:.4f} arcsec")

    offset_ax.set_xlim([-(axlim + .2), axlim + .2])
    offset_ax.set_ylim([-(axlim + .2), axlim + .2])
    ra_ax.set_xlim([-(axlim + .2), axlim + .2])
    dec_ax.set_ylim([-(axlim + .2), axlim + .2])

    ra_ax.set_xticks(range(-axlim, axlim+1))
    dec_ax.set_yticks(range(-axlim, axlim+1))
    offset_ax.set_xticks(range(-axlim, axlim+1))
    offset_ax.set_yticks(range(-axlim, axlim+1))
    
    leg1 = offset_ax.legend(handles=handles, labels=labels, loc=3, ncol=4)
    leg2 = offset_ax.legend(handles=[medline, stdline, pixbox],
                            labels=['median', 'stddev', 'single pixel'],
                            loc=1)
    offset_ax.add_artist(leg1)
    offset_ax.add_artist(leg2)
    
    if save:
        fig.savefig(f'{save}.png', dpi=300, format='png', bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    main()
