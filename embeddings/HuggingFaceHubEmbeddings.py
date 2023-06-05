from langchain.embeddings import HuggingFaceEmbeddings


class HuggingFaceHubEmbeddings:
    def __init__(self):
        self.hf_embeddings = HuggingFaceEmbeddings()

    def embed_query(self, input):
        return self.hf_embeddings.embed_query(input)

    def embed_documents(self, input):
        return self.hf_embeddings.embed_documents(input)


if __name__ == '__main__':
    hf_hub_embeddings = HuggingFaceHubEmbeddings()
    print(hf_hub_embeddings.embed_query("Hello, my name is"))
    print(hf_hub_embeddings.embed_documents(["Hello, my name is", "Guliver, I am a giant", "are you there?"]))
