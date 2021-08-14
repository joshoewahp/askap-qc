#!/usr/bin/env python

import logging
import itertools as it
from askap import Image
from catalog import Catalog
from crossmatch import Crossmatch

logger = logging.getLogger(__name__)

def match_cats(files, refcat, maxoffset, isolationlim, snrlim, outdir):

    image = Image(files, refcat=refcat.sources)
    logger.info(f"Crossmatching {image}")

    catalogs = {}

    try:
        catalogs['askap'] = Catalog(image,
                                    survey_name='askap',
                                    isolationlim=isolationlim,
                                    snrlim=snrlim)
    except AssertionError as e:
        logger.debug(e)

    try:
        catalogs[refcat.name] = Catalog(image,
                                        survey_name=refcat.name,
                                        isolationlim=isolationlim,
                                        snrlim=snrlim)
        print(image)
        print(catalogs['racs'].sources)
    except AssertionError as e:
        logger.exception(e)

    # Try to generate the other catalogues in turn
    try:
        catalogs['sumss'] = Catalog(image,
                                    survey_name='sumss',
                                    frequency=843.0e6,
                                    snrlim=snrlim,
                                    isolationlim=isolationlim)

    except AssertionError as e:
        logger.debug(e)

    try:
        catalogs['nvss'] = Catalog(image,
                                   survey_name='nvss',
                                   frequency=1400.0e6,
                                   snrlim=snrlim,
                                   isolationlim=isolationlim)
    except AssertionError as e:
        logger.debug(e)

    try:
        catalogs['icrf'] = Catalog(image, 'icrf')
    except AssertionError as e:
        logger.debug(e)

    cats = [comb for comb in it.combinations(catalogs, 2) if 'askap' in comb]

    # Try pairwise crossmatches between catalogs
    for cat1, cat2 in cats:

        try:
            logger.debug(f"Crossmatching {cat1} with {cat2}.")

            cm = Crossmatch(catalogs[cat1], catalogs[cat2], maxoffset=maxoffset)
            cm.save(outdir)

            logger.debug(f"{len(cm.matches)} {cm.comp_cat.survey_name} matches located in {image.fieldname}.")

        except AssertionError as e:
            logger.debug(e)

    return
