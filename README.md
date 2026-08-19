# 📄 PDF Summarizer — Hugging Face

A local PDF summarization web app built with **Python, Streamlit, Hugging Face Transformers, PyTorch, and PyPDF**.

Upload a text-based PDF and generate a concise summary and key points using an open-source Hugging Face model. No API key is required.

## 🚀 Features

* Upload and summarize PDF documents
* Local Hugging Face model inference
* Supports BART and DistilBART
* Automatic text extraction and chunking
* Hierarchical summarization for long PDFs
* Key-point extraction
* Streamlit web interface
* CPU support

## 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **Hugging Face Transformers**
* **PyTorch**
* **PyPDF**

## 📁 Structure

```text
pdf-summarizer-huggingface/
├── app.py
├── summarizer.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Setup

```bash
git clone https://github.com/YOUR_USERNAME/pdf-summarizer-huggingface.git
cd pdf-summarizer-huggingface
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the Streamlit URL shown in the terminal.

## 🤖 Models

* `facebook/bart-large-cnn` — better quality, slower
* `sshleifer/distilbart-cnn-12-6` — faster, smaller

The model is downloaded automatically on first use and cached locally.

## ⚠️ Limitations

Currently supports **text-based PDFs only**. Scanned/image-only PDFs require OCR.

