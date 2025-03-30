from paddleocr import PaddleOCR

ocr2 = PaddleOCR(
    use_angle_cls=True,
    det_db_box_thresh=0.3,       # Lower threshold to detect faint text
    det_model_dir='models/ch_PP-OCRv3_det_infer',
    rec_model_dir='models/ch_PP-OCRv3_rec_infer',
    rec_image_shape='3, 48, 640',
    lang='en'
)

