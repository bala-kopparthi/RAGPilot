# documents/

This folder contains the source documents ingested by the RAG pipeline.

## What's included (tracked in git)

| File | Type | Description |
|---|---|---|
| `sample_test_case.txt` | Test Cases | E-commerce checkout flow — 3 test cases with bug IDs |
| `bug_report_BUG-117.txt` | Bug Report | Password reset email failure for mixed-case usernames |
| `requirements_REQ-042.txt` | Requirements | Multi-Factor Authentication (MFA) feature specification |
| `test_case_TC-051.txt` | Test Case | Password reset happy path — 9-step end-to-end test |

## Adding your own documents

Drop any `.txt`, `.pdf`, or `.docx` files into this folder and re-run:

```bash
python ingest.py
```

The pipeline will automatically detect and ingest any new files.

## Note on PDFs

Large or copyrighted PDFs (e.g., textbooks) are excluded from git via `.gitignore`.  
Add them locally — they will be ingested but never committed.

> **Known limitation:** `PyPDFLoader` extracts only the text layer of PDFs.  
> Content inside figures, diagrams, and scanned pages will not be retrieved.  
> See the main README for details and proposed solutions.
