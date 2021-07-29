#!/usr/bin/env python

import logging
import itertools as it
from askap import Image
from catalog import Catalog
from crossmatch import Crossmatch

logger = logging.getLogger(__name__)

def match_cats(files, refcat, maxoffset, isolim, snrlim, outdir):

    image = Image(files, refcat=refcat.sources)
    logger.info(f"Crossmatching {image.name}")

    catalogs = {}

    try:
        catalogs[refcat.name] = Catalog(image,
                                        survey_name=refcat.name,
                                        isolim=isolim,
                                        snrlim=snrlim)
    except AssertionError as e:
        logger.kxception(e)

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

            # Skip crossmatches between secondary catalogues
            if cat2 == 'icrf' and cat1 != 'askap':
                continue

            cm = Crossmatch(catalogs[cat1], catalogs[cat2], maxoffset=maxoffset)
            cm.save(outdir)

            logger.debug(f"{len(cm.df)} {cm.comp_cat.name} matches located in {image.name}.")

        except AssertionError as e:
            logger.debug(e)

    return
