from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.llms import OpenAIChat

from dataloader.JavaSourceProcessor import JavaSourceProcessor
from embeddings.HuggingFaceHubEmbeddings import HuggingFaceHubEmbeddings
from index.ChromaVectoriser import ChromaVectoriser
from index.DataRetriever import DataRetriever
from index.DeepLakeVectoriser import DeepLakeVectoriser

if __name__ == '__main__':
    source_processor = JavaSourceProcessor()
    docs = source_processor.load_source('/Users/centaurus/development/ai/data/jdk')
    print(f'loaded docs - {len(docs)}')
    tokenised_docs = source_processor.split_java_files(docs)
    embeddings = HuggingFaceEmbeddings()
    vector_db = DeepLakeVectoriser().vectorise('./deep_lake', tokenised_docs, embeddings)
    print('created vector db')
    data_retriever = DataRetriever()
    retriever = data_retriever.make_retriever(vector_db)

    print('created retriever from vector db')
    print(vector_db.similarity_search('Map<K,V>', k=5))
    # model = ChatOpenAI(model_name='gpt-3.5-turbo')  # switch to 'gpt-4'
    # qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)
    #
    # # quizz = ['stream multi groupby']
    # chat_history = []
    # question = 'how many map implementations are there?'
    # print('Now answering question')
    # result = qa({"question": question, "chat_history": chat_history})
    # chat_history.append((question, result['answer']))
    # print(f"-> **Question**: {question} \n")
    # print(f"**Answer**: {result['answer']} \n")

    # vector_db.force_delete_by_path('./deep_lake')
