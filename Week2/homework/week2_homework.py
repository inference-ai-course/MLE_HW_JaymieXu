from PIL import Image
import pytesseract  # pip install pytesseract first

# Load an image using Pillow (PIL)
image = Image.open('image.png')

# Perform OCR on the image
text = pytesseract.image_to_string(image)

print(text)
