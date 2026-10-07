"""Small deterministic checks; no model download or GPU required."""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image

from color_benchmark import evaluate, parse_colors, prepare, query_set, rrf_scores, title_colors


class ColorTests(unittest.TestCase):
    def test_aliases_multicolor_and_word_boundaries(self):
        self.assertEqual(title_colors('Navy Blue & Grey shirt'), ('blue', 'gray'))
        self.assertEqual(title_colors('Pink and Black dress'), ('black', 'pink'))
        self.assertEqual(title_colors('Off White shirt'), ('white',))
        self.assertEqual(title_colors('Redwood brand multicolour'), ())
        with self.assertRaises(ValueError):
            parse_colors('blu')

    def test_full_rrf_excludes_self_and_ties_are_stable(self):
        scores = rrf_scores(np.array([99., 3., 2., 1.]), np.array([99., 1., 3., 2.]),
                            np.array([False, True, True, True]))
        self.assertEqual(scores[0], 0)
        self.assertAlmostEqual(scores[2], 1/62 + 1/61)
        self.assertEqual(int(np.argmax(scores)), 2)

    def test_filter_missing_slots_and_exclude_same_product(self):
        products = pd.DataFrame(dict(product_id=['a', 'a', 'b', 'c'],
                                     category=['shirt']*4, color_tags=['red', 'red', 'red', 'blue']))
        matrix = np.array([[[100., 99., 2., 3.]]]*4)
        queries = [dict(index=0, color='red', color_index=0)]
        base, _ = evaluate(matrix, products, queries, [1, 5])
        filtered, detail = evaluate(matrix, products, queries, [1, 5], True)
        self.assertEqual(base[1]['hit_rate'], 0)
        self.assertEqual(filtered[1]['hit_rate'], 1)
        self.assertEqual(filtered[5]['precision'], .5)
        self.assertEqual(filtered[5]['recall'], 1)
        self.assertEqual(detail[0]['result_ids'], 'b')

    def test_color_match_alone_is_not_category_relevance(self):
        products = pd.DataFrame(dict(product_id=['a', 'b', 'c'],
                                     category=['shirt', 'shirt', 'shoe'], color_tags=['red']*3))
        matrix = np.array([[[100., 1., 2.]]]*3)
        result, _ = evaluate(matrix, products, [dict(index=0, color='red', color_index=0)], [1])
        self.assertEqual(result[1]['precision'], 0)
        self.assertEqual(result[1]['color_precision'], 1)

    def test_query_coverage_and_no_product_name_leak(self):
        products = pd.DataFrame(dict(product_id=['a', 'b', 'c'],
                                     category=['shirt', 'shirt', 'shoe'], color_tags=['red']*3))
        queries, skipped = query_set(products)
        self.assertEqual(len(queries), 2)
        self.assertEqual(skipped, 1)
        self.assertEqual(queries[0]['text'], 'red shirt')

    def test_reviewed_labels_override_title_and_exclude_pending(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for i in range(3):
                Image.new('RGB', (10+i, 10), 'blue').save(root / f'{i}.png')
            pd.DataFrame(dict(product_id=['a', 'b', 'c'], product_name=['Red shirt']*3,
                              category=['shirt']*3, split=['test']*3,
                              image_path=[f'{i}.png' for i in range(3)])).to_csv(root / 'data.csv', index=False)
            pd.DataFrame(dict(product_id=['a', 'b', 'c'], color_tags=['blue']*3,
                              reviewed=['true', 'true', 'false'])).to_csv(root / 'labels.csv', index=False)
            output = root / 'result'
            output.mkdir()
            products, original_count = prepare(SimpleNamespace(csv=root/'data.csv', image_dir=root,
                                                labels=root/'labels.csv', split='test'), output)
            self.assertEqual(original_count, 3)
            self.assertEqual(set(products.product_id), {'a', 'b'})
            self.assertEqual(set(products.color_tags), {'blue'})
            self.assertEqual(set(products.color_source), {'human_reviewed'})
            self.assertTrue((output / 'color_review.html').is_file())


if __name__ == '__main__':
    unittest.main()
