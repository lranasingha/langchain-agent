class DataRetriever:

    def make_retriever(self, vector_db, distance_metric='cos', fetch_k=100, k=10):
        retriever = vector_db.as_retriever()
        retriever.search_kwargs['distance_metric'] = distance_metric
        retriever.search_kwargs['fetch_k'] = fetch_k
        retriever.search_kwargs['k'] = k
        retriever.search_kwargs['maximal_marginal_relevance'] = True
        return retriever
