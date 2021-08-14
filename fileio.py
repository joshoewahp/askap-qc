import pandas as pd
from astropy.table import Table
from pathlib import Path

def load_selavy_file(selavypath: Path) -> pd.DataFrame:
    """Import selavy catalogue to pandas DataFrame."""

    if isinstance(selavypath, str):
        selavypath = Path(selavypath)

    # Handle loading of multiple source file formats
    if selavypath.suffix in ['.xml', '.vot']:
        sources = Table.read(
            selavypath, format="votable", use_names_over_ids=True
        ).to_pandas()
    elif selavypath.suffix == '.csv':
        # CSVs from CASDA have all lowercase column names
        sources = pd.read_csv(selavypath).rename(
            columns={"spectral_index_from_tt": "spectral_index_from_TT"}
        )
        # Remove unused columns for consistency
        sources.drop(columns=['id', 'catalogue_id', 'first_sbid', 'other_sbids',
                              'project_id', 'quality_level', 'released_date'],
                     inplace=True)
    else:
        sources = pd.read_fwf(selavypath, skiprows=[1]).drop(columns=['#'])

    return sources
