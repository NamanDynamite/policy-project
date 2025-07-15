import os
import uuid
import datetime
import json
import re
from typing import List, Dict, Any
from langchain_core.runnables import RunnableSequence
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, validator
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from PyPDF2 import PdfReader
from docx import Document

app = FastAPI(
    title="Invoice Reimbursement Analyzer",
    description="API for analyzing employee invoices against company policy"
)

class Config:
    CHROMA_DIR = "chroma_store"
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    CAB_ALLOWANCE_CAP = 150.0
    MEAL_ALLOWANCE_CAP = 200.0
    TRIP_ALLOWANCE_CAP = 2000.0

class ChatQuery(BaseModel):
    query: str

    @validator('query')
    def validate_query(cls, v):
        if len(v.strip()) < 3:
            raise ValueError("Query must be at least 3 characters long")
        return v.strip()

def initialize_services():
    embedding_model = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)
    vector_store = Chroma(persist_directory=Config.CHROMA_DIR, embedding_function=embedding_model)
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY not found in environment variables")
    llm = ChatGroq(api_key=groq_api_key, model_name="llama3-8b-8192", temperature=0)
    return vector_store, llm

vector_store, llm = initialize_services()


CHATBOT_PROMPT = PromptTemplate(
    input_variables=["context", "query"],
    template="""
You are a reimbursement assistant.

Use ONLY the context below to answer. DO NOT assume or invent anything.

If asked for a summary, output a markdown table with:
- Invoice ID
- Employee Name
- Reimbursement Status
- Amount Eligible
- Brief Reason

Context:
{context}

Query:
{query}
"""
)

def extract_text_from_file(file: UploadFile) -> str:
    file.file.seek(0)
    if file.content_type == "application/pdf":
        reader = PdfReader(file.file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    elif file.content_type.endswith("document.wordprocessingml.document"):
        doc = Document(file.file)
        return "\n".join([para.text for para in doc.paragraphs])
    raise ValueError("Unsupported file type")

def normalize_text(text: str) -> str:
    text = re.sub(r"(?<=[A-Za-z])\s+(?=[A-Za-z])", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower().replace(",", "").replace("inr", "₹")

def extract_amounts(text: str) -> Dict[str, float]:
    amounts = {}
    if 'cab' in text or 'ridefee' in text or 'driver' in text:
        for match in re.findall(r'(?:ridefee|fare|cabfare)\s*[:\-₹]?\s*(\d+(?:\.\d+)?)', text):
            amt = float(match)
            if 10 <= amt <= 500:
                amounts['cab_fare'] = max(amounts.get('cab_fare', 0), amt)
    if any(k in text for k in ['biryani', 'whisky', 'restaurant', 'meal', 'tableno', 'dosa', 'idli', 'chapati', 'receipt']):
        meal_amounts = []
        patterns = [
            r'(?:sub\s*total|subtotal|sub\s+total)\s*[:\-]?\s*₹?\s*(\d{2,5}(?:\.\d+)?)',
            r'(?:total|grand\s*total)\s*[:\-]?\s*₹?\s*(\d{2,5}(?:\.\d+)?)',
            r'₹\s*(\d{2,5}(?:\.\d+)?)\s*(?:total|subtotal|sub\s+total)',
            r'(\d{2,5}(?:\.\d+)?)\s*(?:total|subtotal|sub\s+total)',
        ]
        for pat in patterns:
            matches = re.findall(pat, text, flags=re.IGNORECASE)
            for match in matches:
                try:
                    amt = float(match)
                    if 50 <= amt <= 2000:
                        meal_amounts.append(amt)
                except ValueError:
                    continue
        if meal_amounts:
            amounts['meal'] = max(meal_amounts)
    if any(k in text for k in ['ticket', 'bus', 'flight', 'airlines', 'train']):
        for match in re.findall(r'(?:totalfare|netamount|amountpaid|totalpayable|fare)[\s:\-₹]*([\d,]{4,7}(?:\.\d+)?)', text):
            amt = float(match.replace(",", ""))
            if 50 <= amt <= 20000:
                amounts['trip'] = max(amounts.get('trip', 0), amt)
    return amounts

def validate_reimbursement(amount: float, category: str) -> Dict[str, Any]:
    cap = {
        'cab_fare': Config.CAB_ALLOWANCE_CAP,
        'meal': Config.MEAL_ALLOWANCE_CAP,
        'trip': Config.TRIP_ALLOWANCE_CAP
    }.get(category, 0)
    if amount <= cap:
        return {
            "reimbursement_status": "Fully Reimbursed",
            "reason": f"The {category.replace('_', ' ')} of ₹{amount:.2f} is within the policy limit of ₹{cap}.",
            "amount_eligible": amount
        }
    else:
        return {
            "reimbursement_status": "Partially Reimbursed",
            "reason": f"The {category.replace('_', ' ')} of ₹{amount:.2f} exceeds the policy limit of ₹{cap}. Only ₹{cap} will be reimbursed.",
            "amount_eligible": cap
        }

def analyze_invoice_with_policy(policy_text: str, invoice_text: str) -> Dict[str, Any]:
    employee_name = extract_employee_name(invoice_text)
    normalized = normalize_text(invoice_text)

    amounts = extract_amounts(normalized)

    if not amounts:
        return {
            "reimbursement_status": "Declined",
            "reason": "No reimbursable category detected in invoice.",
            "amount_eligible": 0,
            "categories": {}
        }
    results = {cat: validate_reimbursement(amt, cat) for cat, amt in amounts.items()}

    total = sum(r["amount_eligible"] for r in results.values())
    reason = " | ".join(r["reason"] for r in results.values())
    status = "Fully Reimbursed" if all(r["reimbursement_status"] == "Fully Reimbursed" for r in results.values()) else "Partially Reimbursed"
    return {
        "reimbursement_status": status,
        "reason": reason,
        "amount_eligible": total,
        "categories": results
    }

def extract_employee_name(text: str) -> str:
    match = re.search(r'(?:Customer\s*Name|Cust\s*Name)\s*[:\-]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r'Passenger\s+Details.*?:?\s*([A-Z][a-z]+)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r'\bName\b[:\-]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    lines = text.splitlines()
    for line in lines:
        if "Name" in line and not any(x in line.lower() for x in ["address", "date", "price", "amount", "table"]):
            parts = line.split()
            for i, token in enumerate(parts):
                if token.lower() == "name" and i + 1 < len(parts):
                    name_candidate = parts[i + 1].strip(":")
                    if name_candidate.istitle():
                        return name_candidate
    return "Unknown"

def delete_entries_by_employee(employee_name: str):
    if employee_name == "Unknown":
        return
    all_docs = vector_store.similarity_search("invoice", k=100)
    for doc in all_docs:
        if doc.metadata.get("employee_name", "").lower() == employee_name.lower():
            invoice_id = doc.metadata.get("invoice_id")
            if invoice_id:
                vector_store.delete([invoice_id])

@app.post("/analyze_invoice", response_model=Dict[str, Any])
async def analyze_invoice(policy_file: UploadFile = File(...), invoice_files: List[UploadFile] = File(...)):
    try:
        if not policy_file.filename.endswith('.docx'):
            raise HTTPException(status_code=400, detail="Policy file must be a DOCX document")
        policy_text = extract_text_from_file(policy_file)

        results = []
        for invoice_file in invoice_files:
            if not invoice_file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Invoice files must be PDF documents")
            invoice_text = extract_text_from_file(invoice_file)

            employee_name = extract_employee_name(invoice_text)
            employee_name = re.sub(r'\b(Mobile|Number|Phone|Address)\b.*', '', employee_name, flags=re.IGNORECASE).strip()

            delete_entries_by_employee(employee_name) 
 
            invoice_id = str(uuid.uuid4())
            today = datetime.date.today().isoformat()
            analysis_result = analyze_invoice_with_policy(policy_text, invoice_text)

            document_metadata = {
                "invoice_id": invoice_id,
                "employee_name": employee_name,
                "employee_name_search": employee_name.lower(),
                "date": today,
                "status": analysis_result["reimbursement_status"],
                "reason": analysis_result["reason"],
                "amount_eligible": analysis_result["amount_eligible"]
            }
            document_text = (
                f"Invoice ID: {invoice_id}\n"
                f"Employee: {employee_name}\n"
                f"Status: {analysis_result['reimbursement_status']}\n"
                f"Amount Eligible: ₹{analysis_result['amount_eligible']}\n"
                f"Reason: {analysis_result['reason']}\n"
            )
            vector_store.add_texts(
                texts=[document_text],
                metadatas=[document_metadata],
                ids=[invoice_id]
            )

            results.append({
                "invoice_id": invoice_id,
                "status": analysis_result["reimbursement_status"],
                "reason": analysis_result["reason"],
                "amount_eligible": analysis_result["amount_eligible"]
            })
        return {"success": True, "results": results}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query_chatbot")
async def query_chatbot(input: ChatQuery):
    try:
        if not input.query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        all_docs = vector_store.similarity_search("invoice", k=100)


        sorted_docs = sorted(
            all_docs,
            key=lambda d: d.metadata.get("date", ""),
            reverse=True
        )



        recent_docs = sorted_docs[:3] 


        context = "\n".join([
            f"- Invoice ID: {doc.metadata.get('invoice_id', 'N/A')}\n"
            f"  Employee: {doc.metadata.get('employee_name', 'Unknown')}\n"
            f"  Status: {doc.metadata.get('status')}\n"
            f"  Amount Eligible: ₹{doc.metadata.get('amount_eligible')}\n"
            f"  Reason: {doc.metadata.get('reason')}\n"
            for doc in recent_docs
        ])


        
        chatbot_chain = CHATBOT_PROMPT | llm

        response_obj = chatbot_chain.invoke({"context": context, "query": input.query})

        response_text = response_obj.content if hasattr(response_obj, "content") else str(response_obj)

        return {"success": True, "response": response_text}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
