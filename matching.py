import logging
import itertools as it
from askap import Image
from catalog import Catalog
from crossmatch import Crossmatch

logger = logging.getLogger(__name__)

def match_cats(files, refcat, maxoffset, isolim, snrlim):

    image = Image(files, refcat=refcat)
    logger.info(f"Crossmatching {image.name}")

    if 'RACS' in refcat:
        refname = 'racs'
    else:
        refname = 'ref'

    catalogs = {}

    try:
        catalogs[refname] = Catalog(image,
                                    survey_name=refname,
                                    isolim=isolim,
                                    snrlim=snrlim)
    except AssertionError as e:
        logger.debug(e)

    try:
        catalogs['askap'] = Catalog(image,
                                    survey_name='askap',
                                    isolim=isolim,
                                    snrlim=snrlim,
                                    selavy_path=files.selavy)

    except AssertionError as e:
        logger.debug(e)

    # Try to generate the other catalogues in turn
    try:
        catalogs['sumss'] = Catalog(image,
                                    survey_name='sumss',
                                    frequency=843.0e6,
                                    snrlim=snrlim,
                                    isolim=isolim)

    except AssertionError as e:
        logger.debug(e)

    try:
        catalogs['nvss'] = Catalog(image,
                                   survey_name='nvss',
                                   frequency=1400.0e6,
                                   snrlim=snrlim,
                                   isolim=isolim)
    except AssertionError as e:
        logger.debug(e)

    try:
        catalogs['icrf'] = Catalog(image, 'icrf')
    except AssertionError as e:
        logger.debug(e)

    crossmatches = {}
    cats = [comb for comb in it.combinations(catalogs, 2) if 'askap' in comb or 'icrf' in comb]

    # Try pairwise crossmatches between catalogs
    for cat1, cat2 in cats:

        try:
            logger.debug(f"Crossmatching {cat1} with {cat2}.")
            if cat1 == 'icrf':
                cm = Crossmatch(catalogs[cat2], catalogs[cat1], maxsep=maxoffset, scale_flux=False)
            elif cat2 == 'icrf':
                if cat1 != 'askap':
                    continue
                cm = Crossmatch(catalogs[cat1], catalogs[cat2], maxsep=maxoffset, scale_flux=False)
            else:
                cm = Crossmatch(catalogs[cat1], catalogs[cat2], maxsep=maxoffset)

            logger.debug(f"{len(cm.df)} {cat2} matches located in field.")
            crossmatches[f'{cat1}-{cat2}'] = cm

        except AssertionError as e:
            logger.debug(e)

    return crossmatches
