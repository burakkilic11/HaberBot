---

# Project Report: Agentic AI Chatbot Focused on Official Gazette and General News

## 1. Introduction

This report describes the development and architecture of an agentic artificial intelligence chatbot system designed to analyze user queries, determine the relevant information source (Official Gazette or general news), and generate an appropriate response. The project is built upon a supervisor-agent architecture using the LangGraph framework, features a Streamlit interface for user interaction, and is packaged with Docker for ease of deployment. The primary goal of the system is to automatically select the most suitable information source based on the nature of the user's question (Official Gazette database or current news API) and produce the relevant answer.

## 2. System Architecture

The developed system relies on a **supervisor-agent architecture** modeled using LangGraph. In this architecture, a central "supervisor" node analyzes the incoming request and directs the task to the appropriate "agent" node.

*   **Supervisor Node (`classify_question_node`):**
    *   Receives the incoming user question.
    *   Utilizes a pre-trained Large Language Model (LLM - `deepseek-r1:14b`) to classify the question into one of three categories:
        *   `resmi_gazete`: Questions related to the content of the Republic of Turkey Official Gazette.
        *   `general`: Questions about current events, general knowledge, or topics outside the Official Gazette.
        *   `irrelevant`: Meaningless, inappropriate, or unanswerable questions.
    *   Writes the classification result to the LangGraph state (`AgentState`). This state is used to carry information between other nodes in the workflow.

*   **Agent Nodes:** Based on the classification determined by the supervisor, one of the following agent nodes is triggered:
    1.  **Official Gazette RAG Agent (`resmi_gazete_rag_node`):**
        *   Handles questions classified as `resmi_gazete`.
        *   Performs a semantic search (Retrieval) using the user's question on vector representations of Official Gazette documents, which were generated using the `bge-m3:latest` embedding model and stored in ChromaDB.
        *   Retrieves the most relevant document chunks (context).
        *   Presents this context and the original question to the LLM (`deepseek-r1:14b`), instructing it to generate an answer based *solely on the provided context* (Retrieval-Augmented Generation - RAG).
        *   Writes the generated answer, the source ("Official Gazette"), and the used context to the state.
    2.  **General Knowledge Agent (`general_knowledge_node`):**
        *   Handles questions classified as `general`.
        *   **Priority:** If the `NEWSDATA_API_KEY` environment variable is defined, it searches for current Turkish news related to the question using the NewsData.io API.
        *   **If News Found:** Formats the titles and summaries of the found news articles as context. Presents this news context and the original question to the LLM, asking it to generate an answer utilizing both the news and its own general knowledge. The source is specified as "General Knowledge (NewsData.io)".
        *   **If News Not Found / API Error / No API Key:** Skips the news search step or notes the error. Directly asks the question to the LLM, requesting it to generate an answer based on its internal knowledge. The source is specified as "General Knowledge (LLM)" or indicates the relevant error status.
        *   Writes the generated answer and the source to the state. (This agent does not return RAG context).
    3.  **Fallback Agent (`fallback_node`):**
        *   Handles questions classified as `irrelevant`.
        *   Generates a standard, polite message stating that it cannot answer the question.
        *   Writes the answer and the source ("No Answer") to the state.

*   **Routing Logic (`route_question`):** Executes after the supervisor node. It looks at the `classification` value in the state and directs the workflow to the corresponding agent node (`resmi_gazete_agent`, `general_agent`, `fallback_agent`).

*   **State Management (`AgentState`):** LangGraph's `TypedDict`-based state carries information such as `question`, `classification`, `context` (for Official Gazette RAG), `answer`, `source`, and `error` between nodes and contains the final result at the end of the workflow.

*   **Termination (`END`):** After each agent node completes its task, the workflow ends.

## 3. Technologies and Tools Used

*   **Streamlit:** Used to create a simple and interactive web interface allowing users to ask questions and view the chatbot's responses (including source and RAG context details). Chat history is maintained in `st.session_state`.
*   **LangGraph:** The core framework used to implement the supervisor-agent architecture, manage state, and enable conditional routing between nodes. The workflow was created by defining a state machine (`StateGraph`).
*   **LangChain:** A library that integrates with LangGraph and facilitates interaction with LLMs, embedding models, and vector databases. Specifically, `ChatOllama`, `OllamaEmbeddings`, and `Chroma` (vector store interface) components were used.
*   **Ollama:** Used to run and serve Large Language Models (LLM - `deepseek-r1:14b`) and Embedding Models (`bge-m3:latest`) locally. The code is configured to support running Ollama both locally and from within a Docker container (accessing the host machine via the `OLLAMA_BASE_URL` setting).
*   **ChromaDB:** An open-source vector database used to store embeddings (vectors) of text chunks extracted from Official Gazette documents and to perform semantic search (similarity search). The database files (`chroma_db` folder) are stored in the project directory and loaded when the application starts.
*   **NewsData.io API:** An external API used to fetch current news for general knowledge questions. Integration is provided via the `NewsDataApiClient` library. The API key (`NEWSDATA_API_KEY`) must be set as an environment variable; otherwise, this feature is disabled.
*   **Python:** The primary programming language for the project.
*   **Docker:** Used to package the application with all its dependencies, ensuring portability and easy reproducibility across different environments. The `Dockerfile` installs necessary system libraries (e.g., `build-essential` for `chroma-hnswlib`), Python dependencies, copies the application code and ChromaDB data, and starts the Streamlit application.

## 4. Application Flow (Step-by-Step)

1.  **Initialization:** When the application (`app.py`) starts:
    *   Environment variables (`.env`) are loaded (API Keys, Ollama Host).
    *   The Ollama server address is determined (Local/Docker).
    *   Ollama Embedding and LLM models are initialized via the specified address.
    *   The ChromaDB database (`./chroma_db`) is loaded, and a retriever object is created.
    *   The presence of the NewsData.io API key is checked.
    *   The LangGraph workflow (`app`) is compiled.
    *   The Streamlit interface is launched.
2.  **User Interaction:**
    *   The user enters a question through the Streamlit interface.
    *   The question is added to the chat history and displayed on the screen.
3.  **LangGraph Workflow:**
    *   The user's question is passed as input `{"question": prompt}` to the LangGraph `app.invoke()` function.
    *   **Supervisor Node (`classify_question_node`):** Receives the question, classifies it using the LLM (`resmi_gazete`, `general`, `irrelevant`).
    *   **Routing (`route_question`):** Directs the flow to the appropriate agent based on the classification.
    *   **Agent Node (the relevant one):**
        *   *Official Gazette Agent:* Performs RAG from ChromaDB, generates an answer with the LLM.
        *   *General Knowledge Agent:* Attempts NewsData.io (if available), generates an answer with the LLM.
        *   *Fallback Agent:* Generates a standard response.
    *   The selected agent updates the state with `answer`, `source`, `context` (if applicable), and `error` (if any), completing the workflow.
4.  **Response Display:**
    *   The `answer` from the final state returned by LangGraph is displayed to the user.
    *   The `source` information (Official Gazette, General Knowledge (NewsData.io), General Knowledge (LLM), No Answer, Error) is indicated below the response.
    *   If the source is "Official Gazette" and `context` is present, this RAG context is shown in an expandable section under "Detail: RAG Context".
    *   Potential errors (`error`), if any, are reported to the user via `st.error`.
    *   The assistant's response (including all details) is added to the chat history.

## 5. Dockerization

The `Dockerfile` creates a container image for the application by performing the following steps:

1.  Uses the `python:3.11-slim` base image.
2.  Sets the `RUNNING_IN_DOCKER=true` environment variable (for the Ollama host logic within the application).
3.  Installs build tools like `build-essential`. This is crucial for installing Python packages with C++ extensions, such as `chroma-hnswlib`.
4.  Copies the `requirements.txt` file and installs dependencies using `pip`.
5.  Copies the application code (`app.py`) and the ChromaDB data folder (`chroma_db`) into the container.
6.  Exposes port 8501 for Streamlit.
7.  Sets the command (`CMD`) to run `streamlit run app.py` when the container starts, making the application accessible on all network interfaces (`0.0.0.0`).

---
