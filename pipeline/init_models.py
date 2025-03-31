# pipeline/models.py
from paddleocr import PaddleOCR
from transformers import LayoutLMv3TokenizerFast, LayoutLMv3ForTokenClassification
import torch
from functools import lru_cache

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@lru_cache()
def get_ocr():
    return PaddleOCR(
        use_angle_cls=True,
        det_db_box_thresh=0.3,
        det_model_dir='models/ch_PP-OCRv3_det_infer',
        rec_model_dir='models/ch_PP-OCRv3_rec_infer',
        rec_image_shape='3, 48, 640',
        lang='en'
    )

@lru_cache()
def get_model():
    # Load LayoutLMv3 once
    model_path = "klaw09/layoutlmv3-receipt"
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_path).to(device)
    model.eval()
    return model

@lru_cache()
def get_tokenizer():
    # Load LayoutLMv3 once
    model_path = "klaw09/layoutlmv3-receipt"
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(model_path)
    return tokenizer
