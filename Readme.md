Here’s a `README.md` file tailored for my **Invoice Reimbursement Analyzer** FastAPI project using Groq, ChromaDB, LangChain, and LLaMA3. 
This covers project description, setup, usage, and API endpoints:

Invoice Reimbursement Analyzer API

A FastAPI-powered service that analyzes PDF invoices based on a company reimbursement policy and enables contextual Q&A through a chatbot. 
The system uses LangChain with HuggingFace embeddings, Chroma vector store, and Groq's LLaMA3-8B model for inference.



 Features

-  Upload PDF invoices and DOCX policy files
-  Analyze invoices for eligibility across:
  - Meal Allowance
  - Cab Fare
  - Travel Expense
-  Extract structured insights with reasoning
-  Store results in vector DB with metadata
-  Chatbot (`/query_chatbot`) answers questions using most recent entries



 Tech Stack

- FastAPI – Web API framework  
- LangChain – Prompt templates and chains  
- ChromaDB – Local vector database  
- Groq + LLaMA3-8B – Fast LLM inference  
- HuggingFace Embeddings – MiniLM (all-MiniLM-L6-v2)



 Folder Structure



.
├── main.py             
├── chroma\_store/        
├── requirements.txt
└── README.md


Setup Instructions

1. Clone the repo

git clone https://github.com/my-username/invoice-reimbursement-api.git
cd invoice-reimbursement-api


2. Create a virtual environment


python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


3. Install dependencies


pip install -r requirements.txt


4. Set environment variables

Create a `.env` file:


GROQ_API_KEY=my_groq_api_key


5. Run the FastAPI app


uvicorn main:app --reload


API will be served at: `http://localhost:8000/docs



/analyze_invoice` Endpoint

POST `/analyze_invoice`

Analyze one or more invoice PDFs against the company policy DOCX.

Request (multipart/form-data):

* policy_file: HR policy document (.docx)
* invoice_files: List of invoices (.pdf)

Response:

json
{
  "success": true,
  "results": [
    {
      "invoice_id": "bfa2a...uuid...",
      "status": "Fully Reimbursed",
      "reason": "...",
      "amount_eligible": 200.0
    }
  ]
}
```

---

/query_chatbot` Endpoint

POST /query_chatbot

Query the reimbursement assistant. Uses only **latest 3 invoices** from vector DB for context.

Request:

json
{
  "query": "Summarize the reimbursements"
}

Response:

json
{
  "success": true,
  "response": "Here is the summary in a markdown table:\n\n| Invoice ID | ... |"
}



Example Use Cases

* Was my last invoice reimbursed fully?
* Summarize the recent invoices.
* Why was my cab fare partially reimbursed?



 Notes

* The system clears old vector entries per employee before inserting new ones.
* Only most recent documents are used for /query_chatbot to ensure relevance.
* Employee name extraction uses regex heuristics from text.
