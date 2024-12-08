from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import List
from langchain_core.documents import Document
import os
from chroma_utils import vectorstore

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

output_parser = StrOutputParser()

contextualize_q_system_prompt = (
    "You are an AI assistant for the Department of Technical Education (DTE) Rajasthan, responsible for reformulating user queries into precise, standalone, and self-contained forms optimized for retrieving information. Your focus is strictly on government polytechnic colleges in Rajasthan. "
    "For irrelevant or invalid queries outside the scope of government polytechnic colleges, respond: 'I cannot answer this query'. "
    "Do not include explanations, acknowledgments, or any extra text in your output."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_q_system_prompt = (
    "You are an AI-powered chatbot for the Department of Technical Education (DTE) Rajasthan, tasked with answering queries about government polytechnic colleges in Rajasthan based on provided context. Your responses should be accurate, concise, and professional, adhering to government communication standards. "
    "Respond only to queries related to government polytechnic colleges in Rajasthan, covering topics such as admissions, fees, scholarships, facilities, and placements etc. "
    "If the requested information is unavailable, respond: 'The requested information is not currently available. Please refer to the official DTE Rajasthan website for further assistance.' "
    "For irrelevant or out-of-scope queries, respond: 'I can assist only with queries related to government polytechnic colleges in Rajasthan.' "
    "Maintain a friendly and professional tone. Use bullet points or concise paragraphs for clarity. "
    "Greet users if they greet you, and ask how you can assist them."
    "Do not share information about topics outside the scope of government polytechnic colleges in Rajasthan, such as the state's capital or general history etc."
    "Don't provide information outside the provided context like if there are 43 college in given dataset you should be suggesting those college only not other college which may fall under DTE."
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_q_system_prompt),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

os.environ["GROQ_API_KEY"] = "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"

def get_rag_chain(model="llama3-8b-8192"):
    llm = ChatGroq(model=model)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain
