import unittest
from unittest.mock import MagicMock, patch
import ai_photo_editor
from PIL import Image
import numpy as np


class TestPhotoEditor(unittest.TestCase):
    def test_load_raw_image_success(self):
        # Patch the local reference in the module rather than the global rawpy module
        with patch('ai_photo_editor.rawpy') as mock_rawpy:
            mock_raw = MagicMock()
            mock_raw.postprocess.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_rawpy.imread.return_value.__enter__.return_value = mock_raw
            
            # We also need to make sure np is available for the mock return value
            # Since np might be None in the module, let's import it here
            import numpy as np
            
            result = ai_photo_editor.load_raw_image("test.cr2")
            self.assertIsInstance(result, Image.Image)


    def test_apply_adjustments(self):
        # This will fail because apply_adjustments is not yet implemented
        from PIL import Image
        img = Image.new('RGB', (100, 100))
        recs = {
            "edits": [
                {"type": "exposure", "value": 1.2},
                {"type": "contrast", "value": 1.1}
            ]
        }
        result = ai_photo_editor.apply_adjustments(img, recs)
        self.assertIsInstance(result, Image.Image)

    def test_validate_ai_response(self):
        # This will fail because validate_ai_response is not yet implemented
        valid_response = {
            "title": "Test",
            "description": "Desc",
            "edits": [{"type": "exposure", "value": 1.2, "explanation": "exp"}]
        }
        invalid_response = {
            "title": "Test",
            "edits": "not a list"
        }
        self.assertTrue(ai_photo_editor.validate_ai_response(valid_response))
        self.assertFalse(ai_photo_editor.validate_ai_response(invalid_response))

if __name__ == '__main__':
    unittest.main()
