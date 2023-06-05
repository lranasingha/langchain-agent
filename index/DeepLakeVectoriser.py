from langchain.vectorstores import DeepLake


class DeepLakeVectoriser:
    def vectorise(self, db_location, docs, embeddings):
        deep_lake = DeepLake(dataset_path=db_location, read_only=True, overwrite=False)
        deep_lake.add_documents(docs)
        return deep_lake
