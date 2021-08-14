import os
import subprocess
from unittest import TestCase
from click.testing import CliRunner
from create_matches import main


class CreateMatchesTest(TestCase):

    def tearDown(self):
        if os.path.exists('matches/testoutput'):
            os.system("rm -r matches/testoutput")

    def test_whole_dataset(self):
        runner = CliRunner()
        result = runner.invoke(main, '-d data/testdata -S testoutput -R data/testdata/RACS_catalogue_test.fits')
        assert not result.exception

    def test_single_epoch(self):
        runner = CliRunner()
        result = runner.invoke(main, '-e data/testdata/EPOCH08 -S testoutput -R data/testdata/RACS_catalogue_test.fits')
        assert not result.exception

    def test_single_field(self):
        runner = CliRunner()
        rootpath = 'data/testdata/EPOCH08/COMBINED/'
        image = rootpath + 'STOKESI_IMAGES/VAST_0012+00A.EPOCH08.I.fits'
        selavy = rootpath + 'STOKESI_SELAVY/VAST_0012+00A.EPOCH08.I.selavy.components.txt'
        result = runner.invoke(main, f'-f {image} {selavy} -S testoutput -R data/testdata/RACS_catalogue_test.fits')
        assert not result.exception       

    def test_region_selection(self):
        runner = CliRunner()
        result = runner.invoke(main, '-d data/testdata -r 3 -r 4 -S testoutput -R data/testdata/RACS_catalogue_test.fits')
        assert not result.exception       

    def test_missing_args(self):
        runner = CliRunner()
        result = runner.invoke(main, '-r 3 -r 4 -S testoutput -R data/testdata/RACS_catalogue_test.fits')
        assert result.exception
