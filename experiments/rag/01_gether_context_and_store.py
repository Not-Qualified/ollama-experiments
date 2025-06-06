from re import sub
from langchain_core.documents import Document
from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader

import sqlite3

conn = sqlite3.connect('.documents.db')
c = conn.cursor()

# Create table if not exists
c.execute('''
          CREATE TABLE IF NOT EXISTS html (
                                              id INTEGER PRIMARY KEY AUTOINCREMENT,
                                              url TEXT UNIQUE,
                                              title TEXT,
                                              description TEXT,
                                              html TEXT,
                                              created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                              updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
          )
          ''')
# create function to save the documents sqlite
def save_document(document:Document):


    # Upsert document
    c.execute('''
              INSERT INTO html (url, title, description, html)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                title = excluded.title,
                description = excluded.description,
                html = excluded.html,
                updated_at = CURRENT_TIMESTAMP
                  
              ''', (
        document.metadata.get("source", ""),
        document.metadata.get("title", ""),
        document.metadata.get("description", ""),
        document.page_content.strip()
    ))

    conn.commit()


def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return sub(r"\n\n+", "\n\n", soup.text).strip()


loader = RecursiveUrlLoader(
    "https://python.langchain.com/",
    prevent_outside=True,
    base_url="https://python.langchain.com/",
    # extractor=bs4_extractor,
)

generator_docs = loader.lazy_load()


for doc in generator_docs:
    print(doc.metadata)
    print("-----")
    save_document(doc)



conn.close()