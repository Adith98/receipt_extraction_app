from PIL import Image as img
import torch
import time
from PIL import ImageDraw
from pipeline.init_models import get_ocr, get_tokenizer, get_model

ocr = get_ocr()
tokenizer = get_tokenizer()
model = get_model()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ReceiptImage:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = img.open(image_path).convert('RGB')
        self.width, self.height = self.image.size

class OCRPipeline:
    def __init__(self, image: ReceiptImage):
        self.image = image
        self.ocr_result = ocr.ocr(image.image_path, cls=True)
        self.set_text_and_boxes()

    def set_text_and_boxes(self):
        self.texts = []
        self.bboxes = []

        # format ocr result to box, text
        for result in  self.ocr_result[0]:
            bbox, (text, confidence) = result
            word = text

            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x3 = bbox[2][0]
            y3 = bbox[2][1]

            box = [x1, y1, x3, y3]
            box = self.normalize_bbox(box)

            self.texts.append(word)
            self.bboxes.append(box)

    def normalize_bbox(self, bbox):
        return [
            int(1000 * (bbox[0] / self.image.width)),
            int(1000 * (bbox[1] / self.image.height)),
            int(1000 * (bbox[2] / self.image.width)),
            int(1000 * (bbox[3] / self.image.height)),
        ]

class LLMv3Pipeline:
    def __init__(self, model, tokenizer, image: ReceiptImage):
        self.tokenizer = tokenizer
        self.model = model
        self.image = image


        start = time.time()
        self.ocr = OCRPipeline(image)
        stop = time.time()
        print(f"OCR done in {stop - start:.2f} seconds")
        

        start = time.time()
        self.encode_input()
        stop = time.time()
        print(f"Encoding done in {stop - start:.2f} seconds")
        
        
        start = time.time()
        self.run()
        stop = time.time()
        print(f"Model run in {stop - start:.2f} seconds")

    def encode_input(self):
        self.encoding = self.tokenizer(self.ocr.texts , boxes=self.ocr.bboxes, return_tensors="pt", truncation=True, padding=False)

        self.encoding = {k: v.to(device) for k, v in self.encoding.items()}

        self.tokens = self.tokenizer.convert_ids_to_tokens(self.encoding["input_ids"][0].tolist())
        self.aligned_bboxes = self.encoding["bbox"][0].tolist()

    def run(self):
        with torch.no_grad():
            outputs = self.model(**self.encoding)
        
        self.predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()

def merge_word_bboxes(tokens, aligned_bboxes, predictions):
    word_bboxes = token_reconstruction(tokens, aligned_bboxes, predictions)
    merged_word_bboxes = []
    for word_bbox in word_bboxes:
        if merged_word_bboxes and word_bbox["label"] == merged_word_bboxes[-1]["label"] and word_bbox["bbox"] == merged_word_bboxes[-1]["bbox"]:
            merged_word_bboxes[-1]["word"] += " " + word_bbox["word"]
        else:
            merged_word_bboxes.append(word_bbox)
    
    return merged_word_bboxes

def token_reconstruction(tokens, aligned_bboxes, predictions):
    # Initialize variables
    word_bboxes = []
    current_word = ""
    current_bbox = None
    current_label = None

    # Step 1: Token Reconstruction
    for token, bbox, label in zip(tokens, aligned_bboxes, predictions):
        if token in ["<s>", "</s>", "<pad>"]:
            continue  # Skip special tokens

        if token.startswith("Ġ"):  # Start of a new word
            if current_word:
                word_bboxes.append({"word": current_word, "bbox": current_bbox, "label": current_label})  
            current_word = token.replace("Ġ", "")  # Remove prefix
            current_bbox = bbox  # Assign bounding box
            current_label = model.config.id2label.get(label, label)  # Assign prediction label
        else:
            current_word += token  # Append subword
            current_bbox = bbox  # Maintain bounding box
            current_label = model.config.id2label.get(label, label)  # Maintain label
        
    if current_word:
        word_bboxes.append({"word": current_word, "bbox": current_bbox, "label": current_label})  # Save last word
    
    return word_bboxes

def annotate_labels_onto_image(merged_word_bboxes, image: ReceiptImage):
    width, height = image.width, image.height
    draw = ImageDraw.Draw(image.image)

    for word in merged_word_bboxes:
        label = word['label']
        coordinates = word['bbox']
    
        x1, y1 = coordinates[0], coordinates[1]
        x3, y3 = coordinates[2], coordinates[3]
        box = [x1, y1, x3, y3]
        # denormalize
        box = [int((x/1000) * width) if i % 2 == 0 else int((x/1000) * height) for i, x in enumerate(box)]

        # draw label
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), f"{label}", fill="red")
    
    return image.image

def exec(image_path):
    image = ReceiptImage(image_path)

    pipeline = LLMv3Pipeline(model, tokenizer, image)

    tokens = pipeline.tokens
    aligned_bboxes = pipeline.aligned_bboxes
    predictions = pipeline.predictions

    merged_word_bboxes = merge_word_bboxes(tokens, aligned_bboxes, predictions)

    return merged_word_bboxes


# if __name__ == "__main__":
#     image_path = 'static/op.png'
#     extracted_data = exec(image_path)
#     receipt_image = ReceiptImage(image_path)
#     image = annotate_labels_onto_image(extracted_data, receipt_image)
#     image.save('annotated_image.png')