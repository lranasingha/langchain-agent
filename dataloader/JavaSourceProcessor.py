import os

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter, RecursiveCharacterTextSplitter
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)


class JavaSourceProcessor:

    def __init__(self):
        self.exclusions = ['.git', '.jcheck', '.github', 'make', 'build', 'out', 'target', 'bin', 'classes', 'lib',
                           'test', 'doc', 'demo']

    def load_source(self, source_dir: str):
        docs = []
        for root, dirs, files in os.walk(source_dir, topdown=True):
            dirs[:] = [d for d in dirs if d not in self.exclusions]
            for file in files:
                if file.endswith('.java'):
                    text_loader = TextLoader(os.path.join(root, file), encoding='utf-8')
                    docs.extend(text_loader.load_and_split())
        return docs

    def split_java_files(self, source_docs):
        java_code_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.JAVA, chunk_size=100, chunk_overlap=0
        )
        java_docs = java_code_splitter.split_documents(source_docs)
        return java_docs


if __name__ == '__main__':
    java_source_loader = JavaSourceProcessor()
    docs = java_source_loader.load_source("/Users/centaurus/development/ai/data/jdk")
    print(len(docs))
    print(docs[0])
    java_source_loader.split_java_files(docs)

