import os
import chainlit as cl
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

langfuse = Langfuse()

@cl.on_message
async def main(message: cl.Message):
    """
    This function is called every time a user inputs a message in the UI.
    Args:
        message: The user's message.
    Returns:
        None.
    """

    langfuse_handler = CallbackHandler()

    embeddings = OpenAIEmbeddings(
        model=os.getenv("MODEL_EMBEDDINGS_NAME"),
        base_url=os.getenv("MODELS_BASE_URL"),
        api_key=os.getenv("MODEL_EMBEDDINGS_API_KEY")
    )
    sparse = FastEmbedSparse(model_name=os.getenv("MODEL_SPARSE_NAME"))
    qdrant = QdrantVectorStore.from_existing_collection(
        url=os.getenv("QDRANT_URL"),
        embedding=embeddings,
        sparse_embedding=sparse,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        retrieval_mode=RetrievalMode.HYBRID
    )

    llm = OpenAI(
        model=os.getenv("MODEL_LLM_NAME"),
        base_url=os.getenv("MODELS_BASE_URL"),
        api_key=os.getenv("MODEL_LLM_API_KEY"),
        max_tokens=int(os.getenv("MAX_TOKENS_LLM"))
    )


    retriever = qdrant.as_retriever(
        search_type="similarity",
        search_kwargs={"k": int(os.getenv("QDRANT_NB_CHUNKS_RETRIEVED"))}
    )

    prompt_template = """
        Context:
        {context}

        Question:
        {input}

        Answer in French using only the context above.
        If the answer is not in the context, say: "I can't help you...".
        """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    response = retrieval_chain.invoke(
        {"input": message.content},
        config={"callbacks": [langfuse_handler]}
    )

    trace_id = langfuse_handler.last_trace_id

    await cl.Message(
        content=response.get("answer"),
        actions=[
            cl.Action(name="like", payload={"trace_id": trace_id}, label="👍", tooltip="The answer was useful!"),
            cl.Action(name="dislike", payload={"trace_id": trace_id}, label="👎", tooltip="The answer was useless!"),
        ]
    ).send()

@cl.action_callback("like")
async def like(action):
    langfuse.create_score(
        name="user-feedback",
        value=1,
        trace_id=action.payload.get('trace_id')
    )

@cl.action_callback("dislike")
async def dislike(action):
    langfuse.create_score(
        name="user-feedback",
        value=0,
        trace_id=action.payload.get('trace_id'),
    )
