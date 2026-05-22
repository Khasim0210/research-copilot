"""
Conversational RAG: rewrite follow-up questions using chat history,
then retrieve + answer with the standalone question.
"""

import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from src.rag_chain import get_llm, format_docs
from src.vectorstore import load_vectorstore, VECTORSTORE_DIR
from src.ingest import get_embeddings

load_dotenv()


def build_conversational_chain(vectorstore, k: int = 5):
    """
    Two-stage chain:
    1. Rewrite the follow-up into a standalone question using history.
    2. Retrieve docs for the standalone question and answer with citations.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = get_llm()

    # Stage 1: condense follow-up + history -> standalone question
    condense_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and a follow-up question, rephrase the "
         "follow-up into a standalone question. If no rephrasing is needed, "
         "return the question as-is. Output ONLY the standalone question."),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])
    condense_chain = condense_prompt | llm | StrOutputParser()

    # Stage 2: answer grounded in retrieved docs
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a research assistant. Answer the user's question using ONLY "
         "the provided context. If the answer is not in the context, say "
         "'I cannot answer based on the provided documents.' "
         "Cite the source page number for each claim."),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
    ])

    def route(inputs):
        # If no history, skip condensation
        if not inputs["history"]:
            return inputs["question"]
        return condense_chain.invoke(inputs)

    chain = (
        RunnablePassthrough.assign(
            standalone=RunnableLambda(route)
        )
        | RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["standalone"]))
        )
        | RunnablePassthrough.assign(
            answer=lambda x: (answer_prompt | llm | StrOutputParser()).invoke({
                "context": x["context"],
                "question": x["standalone"],
            })
        )
    )
    return chain


if __name__ == "__main__":
    embeddings = get_embeddings()
    vs = load_vectorstore(embeddings, VECTORSTORE_DIR)
    chain = build_conversational_chain(vs)

    # Simulate a multi-turn chat
    history = []
    turns = [
        "What is the Transformer architecture?",
        "How many layers does its encoder have?",
        "And what about the decoder?",
    ]

    for q in turns:
        print(f"\n{'='*60}\nUser: {q}\n{'='*60}")
        result = chain.invoke({"question": q, "history": history})
        print(f"Assistant: {result['answer']}")
        # Add this turn to the history for next iteration
        history.append(HumanMessage(content=q))
        history.append(AIMessage(content=result["answer"]))