import argparse
import multiprocessing as mp
import os

import cv2
import langid
import numpy as np
import pymupdf
import pytesseract
import tqdm
from langcodes import Language
from PIL import Image


class TextExtractor:
    def __init__(self, pdf_dir, image_dir=None, text_dir=None) -> None:

        self.pdf_dir = pdf_dir
        self.image_dir = image_dir
        self.text_dir = text_dir

        if self.image_dir and not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        if self.text_dir and not os.path.exists(self.text_dir):
            os.makedirs(self.text_dir)

    def _extract_text(self, pdf_filename):

        # load pdf
        doc = pymupdf.open(os.path.join(self.pdf_dir, pdf_filename))

        # convert 1st page of pdf to image
        page = doc[0]
        pix = page.get_pixmap(colorspace="GRAY", dpi=300)

        # preprocess image
        im = np.frombuffer(buffer=pix.samples, dtype=np.uint8).reshape(
            (pix.height, pix.width)
        )
        im = cv2.medianBlur(im, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        im = cv2.morphologyEx(im, cv2.MORPH_ERODE, kernel, iterations=1)
        im = Image.fromarray(im, mode="L")

        # save preprocessed image
        if self.image_dir:
            image_save_path = os.path.join(
                self.image_dir, pdf_filename.replace(".pdf", ".png")
            )
            im.save(image_save_path)

        # rough extraction
        text = pytesseract.image_to_string(im, config="--oem 1")

        # detect language
        detected_lang = langid.classify(text)[0]
        detected_lang = Language(detected_lang).to_alpha3()

        # language-specific extraction
        text = pytesseract.image_to_string(im, lang=detected_lang, config="--oem 1")

        # save extracted text
        if self.text_dir:
            text_save_path = os.path.join(
                self.text_dir, pdf_filename.replace(".pdf", ".txt")
            )
            with open(text_save_path, "w") as outfile:
                outfile.write(text)

        return text

    def run(self):
        pdf_filenames = [f for f in os.listdir(self.pdf_dir) if f.endswith(".pdf")]

        extracted_text = []
        with mp.Pool() as pool:
            for result in tqdm.tqdm(
                pool.imap(self._extract_text, pdf_filenames),
                total=len(pdf_filenames),
            ):
                extracted_text.append(result)

        return extracted_text


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=False)
    parser.add_argument("--text_dir", type=str, required=False)
    args = parser.parse_args()

    extractor = TextExtractor(
        pdf_dir=args.pdf_dir,
        image_dir=args.image_dir,
        text_dir=args.text_dir,
    )

    extracted_text = extractor.run()
