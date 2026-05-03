"""Image preprocessing: deskew, binarize, line segmentation.

Default line detector is doctr (robust on cursive); horizontal projection
profile is a fallback for clean printed pages. Returns line image crops with
bounding boxes for downstream OCR.
"""

# TODO: implement preprocess(image_path) -> list[LineCrop]
