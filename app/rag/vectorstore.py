import time

from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.core.config import get_settings


settings = get_settings()

_embeddings = None
_vectorstore = None


# -----------------------------------------
# Embedding Dimensions
# -----------------------------------------

EMBEDDING_DIMENSIONS = {
    "sentence-transformers/all-minilm-l6-v2": 384,
    "all-minilm-l6-v2": 384,
}


# -----------------------------------------
# Get Embedding Dimension
# -----------------------------------------

def get_embedding_dimension(
    model_name: str | None = None
) -> int:

    name = (
        model_name or settings.embedding_model or ""
    ).strip()

    if not name:
        raise RuntimeError(
            "Embedding model is not configured"
        )

    normalized = name.lower()

    if normalized in EMBEDDING_DIMENSIONS:
        return EMBEDDING_DIMENSIONS[normalized]

    raise ValueError(
        f"Unsupported embedding model: {name}. "
        "Add its dimension to EMBEDDING_DIMENSIONS."
    )


# -----------------------------------------
# Initialize Hugging Face Embeddings
# -----------------------------------------

def get_embeddings():

    global _embeddings

    if _embeddings is None:

        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            encode_kwargs={
                "normalize_embeddings": True
            }
        )

    return _embeddings


# -----------------------------------------
# Initialize Pinecone Index
# -----------------------------------------

def ensure_index():

    if not settings.pinecone_api_key:
        raise RuntimeError(
            "PINECONE_API_KEY is missing"
        )

    desired_dimension = get_embedding_dimension()

    pc = Pinecone(
        api_key=settings.pinecone_api_key
    )

    existing_indexes = [
        x["name"] for x in pc.list_indexes()
    ]

    index_name = settings.pinecone_index_name

    # Check whether the index already exists
    if index_name in existing_indexes:

        index_info = pc.describe_index(
            index_name
        )

        current_dimension = getattr(
            index_info,
            "dimension",
            None
        )

        if (
            current_dimension is None
            and isinstance(index_info, dict)
        ):
            current_dimension = index_info.get(
                "dimension"
            )

        # Prevent accidental deletion of an index
        if current_dimension != desired_dimension:

            raise ValueError(
                f"Pinecone index '{index_name}' has "
                f"dimension {current_dimension}, but "
                f"the embedding model requires "
                f"{desired_dimension}. "
                "Use a different index name with "
                "the correct dimension."
            )

    else:

        # Create a new index
        pc.create_index(
            name=index_name,
            dimension=desired_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
        )

    # Wait for the index to become ready
    while True:

        index_info = pc.describe_index(
            index_name
        )

        if index_info.status["ready"]:
            break

        time.sleep(1)

    return pc.Index(index_name)


# -----------------------------------------
# Initialize Vector Store
# -----------------------------------------

def get_vectorstore():

    global _vectorstore

    if _vectorstore is None:

        index = ensure_index()

        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )

    return _vectorstore


# -----------------------------------------
# Get Retriever
# -----------------------------------------

def get_retriever():

    return get_vectorstore().as_retriever(
        search_kwargs={
            "k": settings.top_k
        }
    )


# -----------------------------------------
# Add Documents
# -----------------------------------------

def add_documents(chunks):

    store = get_vectorstore()

    return store.add_documents(chunks)