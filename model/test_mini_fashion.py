import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from mini_fashion import load_catalog


class MiniFashionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'data/data').mkdir(parents=True)
        (self.root / 'data/processed').mkdir()
        (self.root / 'runs/clip_finetune_3epochs_local').mkdir(parents=True)
        for i in range(3):
            Image.new('RGB', (12+i, 12), 'blue').save(self.root / f'data/data/{i}.png')
        self.frame = pd.DataFrame(dict(product_id=['a', 'b', 'c'], product_name=['Blue shirt']*3,
                                      category=['Shirts']*3, image_reference=[f'{i}.png' for i in range(3)],
                                      image_path=[f'Z:\\old-machine\\{i}.png' for i in range(3)],
                                      split=['train', 'validation', 'test']))
        self.frame.to_csv(self.root / 'data/processed/products.csv', index=False)
        self.frame.to_csv(self.root / 'runs/clip_finetune_3epochs_local/dataset.csv', index=False)

    def test_relocation_aliases_and_duplicate_names_keep_ids(self):
        frame = load_catalog(root=self.root)
        self.assertEqual(len(frame), 3)
        self.assertEqual(frame.productDisplayName.tolist(), frame.product_name.tolist())
        self.assertEqual(frame.articleType.tolist(), frame.category.tolist())
        self.assertTrue(all(Path(p).is_file() for p in frame.image))
        self.assertTrue(frame.baseColour.eq('').all())  # Never invent labels.

    def test_saved_splits_remain_disjoint(self):
        self.assertEqual(load_catalog('train', self.root).product_id.tolist(), ['a'])
        self.assertEqual(load_catalog('test', self.root).product_id.tolist(), ['c'])

    def test_missing_csv_fails_instead_of_downloading_other_dataset(self):
        (self.root / 'data/processed/products.csv').unlink()
        with self.assertRaises(FileNotFoundError):
            load_catalog(root=self.root)

    def test_missing_image_fails_instead_of_silent_drop(self):
        (self.root / 'data/data/0.png').unlink()
        with self.assertRaises(FileNotFoundError):
            load_catalog(root=self.root)


if __name__ == '__main__':
    unittest.main()
