from paddleocr import PaddleOCR
from PIL import ImageDraw, Image
import numpy as np

# Paddleocr supports Chinese, English, French, German, Korean and Japanese
# You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
# to switch the language model in order

ocr = PaddleOCR(
    use_angle_cls=True,
    det_db_box_thresh=0.3,       # Lower threshold to detect faint text
    det_model_dir='models/ch_PP-OCRv3_det_infer',
    rec_model_dir='models/ch_PP-OCRv3_rec_infer',
    rec_image_shape='3, 48, 640',
    lang='en'
)


# ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
img_path = 'static/2.jpg'
result = ocr.ocr(img_path, cls=True)


words = []

# format ocr result to box, text
for result in result[0]:
    word = {"quad": {}, "text": None}
    bbox, (text, confidence) = result
    
    word["quad"]["x1"] = bbox[0][0]
    word["quad"]["y1"] = bbox[0][1]
    word["quad"]["x2"] = bbox[1][0]
    word["quad"]["y2"] = bbox[1][1]
    word["quad"]["x3"] = bbox[2][0]
    word["quad"]["y3"] = bbox[2][1]
    word["quad"]["x4"] = bbox[3][0]
    word["quad"]["y4"] = bbox[3][1]
    word["text"] = text

    words.append(word)


img = Image.open(img_path).convert('RGB')

width, height = img.size

print(width, height)

draw = ImageDraw.Draw(img, "RGBA")

for word in words:
  coordinates = word['quad']

  x1, y1 = coordinates['x1'], coordinates['y1']
  x3, y3 = coordinates['x3'], coordinates['y3']
  box = [x1, y1, x3, y3]
  draw.rectangle(box, outline='black', width=2)

img.show()
img.save(f"static/{img_path[0]}99_ocr.jpg")