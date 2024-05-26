import os
from pinecone import Pinecone, ServerlessSpec
# from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from app.chat.embeddings.openai import embeddings

import os
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

# Now do stuff
if 'docs' not in pc.list_indexes().names():
    pc.create_index(
        name='docs', 
        dimension=1536, 
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

vector_store = PineconeVectorStore.from_existing_index(
    os.getenv("PINECONE_INDEX_NAME"),embeddings
)

# vector_store.as_retriever()

def build_retriever(chat_args, k ):
    search_kwargs = {
        "filter" : {"pdf_id": chat_args.pdf_id},
        "k" : k
    }
    return vector_store.as_retriever(
        search_kwargs=search_kwargs
    )


