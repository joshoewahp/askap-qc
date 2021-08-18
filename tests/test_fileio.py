from unittest import TestCase
from fileio import load_selavy_file
from pathlib import Path

class FileIOTest(TestCase):

    def test_load_selavy_file_txt(self):
        path = Path('tests/data/EPOCH05x/COMBINED/STOKESI_SELAVY/')
        path /= 'VAST_0012+00A.EPOCH05x.I.selavy.components.txt'
        f = load_selavy_file(path)
        self.assertEqual(f.shape, (7682, 38))

    def test_load_selavy_file_csv(self):
        path = Path('tests/data/casda_selavy_file.csv')
        f = load_selavy_file(path)
        self.assertEqual(f.shape, (2274, 38))

    def test_load_selavy_file_xml(self):
        path = Path('tests/data/casda_selavy_file.xml')
        f = load_selavy_file(path)
        self.assertEqual(f.shape, (2274, 38))

    def test_string_path_argument(self):
        path = 'tests/data/casda_selavy_file.xml'
        f = load_selavy_file(path)
        self.assertEqual(f.shape, (2274, 38))
