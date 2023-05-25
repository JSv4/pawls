from typing import List, Tuple, Dict
import csv
import io

import numpy
import numpy as np
import pytesseract
import pandas as pd
import cv2
from pdf2image import convert_from_path

from pawls.preprocessors.model import Token, PageInfo, Page
from pawls.commands.utils import get_pdf_pages_and_sizes


def calculate_image_scale_factor(pdf_size, image_size):
    pdf_w, pdf_h = pdf_size
    img_w, img_h = image_size
    scale_w, scale_h = pdf_w / img_w, pdf_h / img_h
    return scale_w, scale_h

def apply_threshold(img, argument):
    switcher = {
        1: cv2.threshold(cv2.GaussianBlur(img, (9, 9), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        2: cv2.threshold(cv2.GaussianBlur(img, (7, 7), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        3: cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        4: cv2.threshold(cv2.medianBlur(img, 5), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        5: cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        6: cv2.adaptiveThreshold(cv2.GaussianBlur(img, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
        7: cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
    }
    return switcher.get(argument, "Invalid method")

def extract_page_tokens(
    pdf_image: "PIL.Image", pdf_size=Tuple[float, float], language="eng"
) -> List[Dict]:

    open_cv_image = numpy.array(pdf_image.convert('RGB'))
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    img = cv2.resize(open_cv_image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = apply_threshold(img, 1)

    _data = pytesseract.image_to_data(img, lang=language)

    scale_w, scale_h = calculate_image_scale_factor(pdf_size, pdf_image.size)

    res = pd.read_csv(
        io.StringIO(_data), quoting=csv.QUOTE_NONE, encoding="utf-8", sep="\t"
    )

    # An implementation adopted from https://github.com/Layout-Parser/layout-parser/blob
    # /20de8e7adb0a7d7740aed23484fa8b943126f881/src/layoutparser/ocr.py#L475
    res_without_na_text_rows = res[~res.text.isna()]

    if res_without_na_text_rows.empty:
        return []

    tokens = (
        res_without_na_text_rows
        .groupby(["page_num", "block_num", "par_num", "line_num", "word_num"], group_keys=False)
        .apply(
            lambda gp: pd.Series(
                [
                    gp["left"].min(),
                    gp["top"].min(),
                    gp["width"].max(),
                    gp["height"].max(),
                    gp["conf"].mean(),
                    gp["text"].astype(str).str.cat(sep=" "),
                ]
            )
        )
        .reset_index(drop=True)
        .reset_index()
        .rename(
            columns={
                0: "x",
                1: "y",
                2: "width",
                3: "height",
                4: "score",
                5: "text",
                "index": "id",
            }
        )
        .drop(columns=["score", "id"])
        .assign(
            x=lambda df: df.x * scale_w,
            y=lambda df: df.y * scale_h,
            width=lambda df: df.width * scale_w,
            height=lambda df: df.height * scale_h,
        )
        .apply(lambda row: row.to_dict(), axis=1)
        .tolist()
    )

    return tokens


def parse_annotations(pdf_file: str) -> List[Page]:

    pdf_images = convert_from_path(pdf_file)
    _, pdf_sizes = get_pdf_pages_and_sizes(pdf_file)
    pages = []
    for page_index, (pdf_image, pdf_size) in enumerate(zip(pdf_images, pdf_sizes)):
        tokens = extract_page_tokens(pdf_image, pdf_size)
        w, h = pdf_size
        page = dict(
            page=dict(
                width=w,
                height=h,
                index=page_index,
            ),
            tokens=tokens,
        )
        pages.append(page)

    return pages


def process_tesseract(pdf_file: str):
    """
    Integration for importing annotations from pdfplumber.
    pdf_file: str
        The path to the pdf file to process.
    """
    annotations = parse_annotations(pdf_file)

    return annotations
