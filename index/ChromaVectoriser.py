from langchain.vectorstores import Chroma


class ChromaVectoriser:
    def vectorise(self, docs, embeddings):
        return Chroma.from_documents(docs, embeddings)
