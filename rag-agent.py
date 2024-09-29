import os
import re
import boto3
from botocore.client import Config
import streamlit as st
import yfinance as yf
from datetime import datetime


from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain.memory import ConversationSummaryMemory
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import MessagesPlaceholder
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.agents import (
    AgentExecutor,
    create_tool_calling_agent,
)
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema import SystemMessage
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_experimental.tools import PythonAstREPLTool
from langchain.tools import tool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# Testing only
# from langchain_groq import ChatGroq
#
#
# llm = ChatGroq(
#     model="llama3-70b-8192",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=3,
#     stop_sequences=[],
# )


session = boto3.session.Session()  # type: ignore
# region = session.region_name
region = "us-west-2"
bedrock_config = Config(
    connect_timeout=120, read_timeout=120, retries={"max_attempts": 0}
)
bedrock_client = boto3.client("bedrock-runtime", region_name=region)

llm = ChatBedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock_client)
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock_client,
    region_name="us-west-2",
)


st.set_page_config(
    page_title="RAG + Agent Code Assistant!",
    page_icon="🖥️",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

st.title("RAG + Agent Code Assistant! 💬")
st.info(
    "This app uses the RAG model to answer questions about finance. Ask me anything! 🏗️",
    icon="💬",
)

system_message = SystemMessage(
    content=(
        f"""You are an agent designed to answer questions about Machine Learning, AI, Finance, AlgoTrading, Stocks, and Quantitative Finance.
Today's Date and Time in DD-MM-YYYY HH:MM:SS is: "{datetime.now().strftime("%d-%m-%Y %H:%M:%S")}". Provide responses in markdown format.
Always use Tools to fetch historical stock data, latest stock price, financial news, run python code and document retriever. If tool is not available then you can write python code to fetch data.
When responding, please provide a clear and complete answer that fully addresses the user's question.
If necessary, provide detailed explanations, examples, and data to support your response.
You can plot charts using matplotlib. If user questions can be explained by chart or python code, please provide the chart or code as a response. Write complete python code without providing hypothetical outputs, use print or plot in code for output.
If you need to run code to get the answer, please do so and include the output in your response. Always use single code block to write python code. Example: ```python\\nprint("Hello, World!")```
Clarity and completeness are more important than conciseness, so please take the time to provide a thorough and accurate response."""
    )
)
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! How can I help you today?",
        }
    ]


def data_ingestion(embeddings):
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    vectorstore_faiss = FAISS.from_documents(docs, embeddings)
    vectorstore_faiss.save_local("faiss_index")


data = st.file_uploader("Upload a PDF file", type=["pdf"])
if data:
    with open(os.path.join("data", data.name), "wb") as f:
        f.write(data.getbuffer())
    with st.spinner(
        "Creating embeddings from the uploaded PDF file. This may take a few minutes."
    ):
        data_ingestion(embeddings)
        st.success("Embeddings created successfully! 🎉")
else:
    pass


def local_faiss_retrieval(embeddings, dirPth="faiss_index"):
    # load saved FAISS vectorstore
    index = FAISS.load_local(dirPth, embeddings, allow_dangerous_deserialization=True)
    return index


# Load and index data
@st.cache_resource(show_spinner=False)
def load_index():
    with st.spinner(
        text="Loading and indexing the building code docs – hang tight! This should take a moment."
    ):
        index = local_faiss_retrieval(embeddings)
        return index


vectorstore = load_index()

# Set up the retriever
retriever = vectorstore.as_retriever()


@tool("fetch_historical_stock_data")
def fetch_historical_stock_data(ticker: str, start_date: str, end_date: str):
    """
    Fetch historical stock data for a given ticker. The start_date and end_date should be in the format 'YYYY-MM-DD'.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


@tool("latest_stock_price")
def latest_stock_price(ticker: str):
    """
    Fetch the latest stock price for a given ticker.
    """
    data = yf.Ticker(ticker)
    return data.history(period="1d")


repl_tool = PythonAstREPLTool()
yf_news_tool = YahooFinanceNewsTool()
retriever_tool = create_retriever_tool(
    retriever,
    "document_retriever",
    "Retrieve relevant financial documents, SEC filings (10k, 10Q), market trends, and other related information based on the user's question. Search for specific terms and keywords mentioned in the user's query to provide accurate and relevant results.",
)

tools = [
    retriever_tool,
    fetch_historical_stock_data,
    latest_stock_price,
    yf_news_tool,
    repl_tool,
]

memory_key = "chat_history"
chat_memory = ConversationSummaryMemory(
    llm=llm,
    memory_key=memory_key,
    return_messages=True,
    output_key="output",
)

prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=system_message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)],
)

agent_engine = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent_engine,
    tools=tools,
    memory=chat_memory,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
    verbose=True,
)


def if_code_block(code):
    pattern = r"```(.*?)```|```python\n(.*?)```|```python(.*?)```"
    code_blocks = re.findall(pattern, code, re.DOTALL)
    if code_blocks:
        code = "".join(code_blocks[0])
        return {"code": code, "is_code_block": True}
    return {"code": None, "is_code_block": False}


def run_code_block(code):
    if code:
        try:
            mod_code = code.replace("plt.show()", "plt.savefig('plot.png')")
            repl_response = repl_tool.run(mod_code)
            if repl_response:
                st.html(
                    f"<big>Code Execution Results:</big><br><code>{repl_response}</code>"
                )
                return repl_response
            if os.path.exists("plot.png"):
                st.image("plot.png", use_column_width=True)
                os.remove("plot.png")
        except Exception as e:
            return f"Error: {e}"
    return None


if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = agent_engine
if "agent_executor" not in st.session_state:
    # Initialize 'agent_executor' and save it to the session state
    st.session_state.agent_executor = agent_executor
if prompt := st.chat_input(
    "Ask your questions here!"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
# Display the prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            st_callback = StreamlitCallbackHandler(st.container())
            response = st.session_state.agent_executor.invoke(
                {"input": prompt}, {"callbacks": [st_callback]}
            )
            st.write(response["output"])
            code_block = if_code_block(response["output"])
            if code_block["is_code_block"]:
                repl_res = run_code_block(code_block["code"])
                message = {
                    "role": "assistant",
                    "content": f"{response['output']}\nCode Execution Result: {repl_res}",
                }
                st.session_state.messages.append(message)
            else:
                message = {
                    "role": "assistant",
                    "content": response["output"],
                }

                # Add response to message history
                st.session_state.messages.append(message)
