from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document


class PDFDataIndex:
    def load(self, pdf_path: str) -> list[Document]:
        pdf_loader = PyPDFLoader(pdf_path)
        pages = pdf_loader.load_and_split()
        return pages
