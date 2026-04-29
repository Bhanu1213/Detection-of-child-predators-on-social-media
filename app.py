from flask import Flask, render_template, request
import os
import csv
import docx
import PyPDF2
from PIL import Image
import pytesseract
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from model.text_model import train_model, predict_text

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model, vectorizer = train_model()

# -------------------------
# TEXT EXTRACTION HELPERS
# -------------------------

def extract_text_from_txt(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.readlines()

def extract_text_from_csv(filepath):
    lines = []
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            lines.extend(row)
    return lines

def extract_text_from_pdf(filepath):
    lines = []
    with open(filepath, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                lines.extend(text.split('\n'))
    return lines

def extract_text_from_docx(filepath):
    doc = docx.Document(filepath)
    return [para.text for para in doc.paragraphs]

def extract_text_from_image(filepath):
    image = Image.open(filepath)
    text = pytesseract.image_to_string(image)
    return text.split('\n')

# -------------------------
# MAIN ROUTE
# -------------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    summary_result = None

    if request.method == 'POST':
        file = request.files.get('file')
        lines = []

        if file:
            filename = file.filename.lower()
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            if filename.endswith('.txt') or filename.endswith('.log'):
                lines = extract_text_from_txt(filepath)

            elif filename.endswith('.csv'):
                lines = extract_text_from_csv(filepath)

            elif filename.endswith('.pdf'):
                lines = extract_text_from_pdf(filepath)

            elif filename.endswith('.docx'):
                lines = extract_text_from_docx(filepath)

            elif filename.endswith(('.png', '.jpg', '.jpeg')):
                lines = extract_text_from_image(filepath)

        predator_count = 0
        non_predator_count = 0
        total_messages = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            label, _, _ = predict_text(model, vectorizer, line)
            total_messages += 1

            if label == "Predator":
                predator_count += 1
            else:
                non_predator_count += 1

        if total_messages > 0:
            predator_percent = round((predator_count / total_messages) * 100, 2)
            non_predator_percent = round((non_predator_count / total_messages) * 100, 2)

            final_decision = (
                "Predator-Dominant Conversation"
                if predator_percent > non_predator_percent
                else "Non-Predator-Dominant Conversation"
            )

            summary_result = {
                "total": total_messages,
                "predator_count": predator_count,
                "non_predator_count": non_predator_count,
                "predator_percent": predator_percent,
                "non_predator_percent": non_predator_percent,
                "final_decision": final_decision
            }

    return render_template("index.html", summary_result=summary_result)

if __name__ == "__main__":
    app.run(debug=True)
