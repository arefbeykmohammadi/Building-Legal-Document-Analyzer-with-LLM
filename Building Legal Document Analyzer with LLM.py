import fitz
import torch
from transformers import AutoTokenizer, AutoModelForPreTraining, pipeline
import spacy
from google.colab import files

uploaded = files.upload()
file_name = list(uploaded.keys())[0]

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = " "
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

pdf_text = extract_text_from_pdf(file_name)
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModelForPreTraining.from_pretrained("nlpaueb/legal-bert-base-uncased")
nlp = spacy.load("en_core_web_sm")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def generate_summary(text):
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]["summary_text"]

def extract_entities(text):
    doc = nlp(text)
    entites = {ent.label_: [] for ent in doc.ents}
    for ent in doc.ents:
        entites[ent.label_].append(ent.text)
    return entites

def extract_clauses(text):
    clauses = {
        "Payment Terms": [],
        "Confidentiality": [],
        "Termination": [],
        "Governing Law": []
    }
    for line in text.split("\n"):
        if "pay" in line.lower() or "compensation" in line.lower():
            clauses["Payment Terms"].append(line)
        elif "confidential" in line.lower() or "disclose" in line.lower():
            clauses["Confidentiality"].append(line)
        elif "terminate" in line.lower():
            clauses["Termination"].append(line)
        elif "law" in line.lower() or "jurisdiction" in line.lower():
            clauses["Governing Law"].append(line)
    return clauses

def analyze_risks(clauses):
    risk_keywords = ["breach", "liability", "penalty", "damages", "termination without cause"]
    risks = []
    for category, clause_list in clauses.items():
        for clause in clause_list:
            if any(word in clause.lower() for word in risk_keywords):
                risks.append(f"Possible Risk in {category}: {clause}")
    return risks

summary = generate_summary(pdf_text)
entities = extract_entities(pdf_text)
clauses = extract_clauses(pdf_text)
risks = analyze_risks(clauses)

print("\n Summary of The Legal Document")
print(summary)

print("\n Named Entities (Parties, Dates, Amounts, ETC)")
for key, values in entities.items():
    if values:
        print(f"{key}: {', '.join(values)}")
print("\n Extracted Clauses")
for key, values in clauses.items():
    if values:
        print(f"\n {key}:")
        for clause in values:
            print(f"-{clause}")
print("\n Risk Analysis")
if risks:
    for risk in risks:
        print(risk)
else:
    print("No high risk clauses detected")