# Agentic Brochure Generator

An AI-powered brochure generator built with Python, OpenAI, LangChain, LangGraph, PyTorch, and RAG.

The project explores how AI agents can research company websites, retrieve relevant information, generate marketing content, and collaborate through an automated editorial workflow.

## Features

- **AI Research Agent:** Discovers and reads relevant company webpages using LangChain and Playwright.
- **RAG & Semantic Search:** Uses local PyTorch-backed embeddings to retrieve relevant information.
- **Brochure Generation:** Generates company brochures in Markdown using OpenAI.
- **Editorial Workflow:** Reviews, scores, and revises generated brochures.
- **Social Media:** Generates LinkedIn posts from approved brochures.
- **LangGraph Orchestration:** Coordinates research, retrieval, writing, review, and revision through a stateful workflow.

## Tech Stack

- Python
- OpenAI API
- LangChain & LangGraph
- PyTorch & Sentence Transformers
- Playwright
- Pydantic
- uv (dependency management)

## Getting Started

**1. Clone the repository**

`git clone https://github.com/YOUR_USERNAME/agentic-brochure-generator.git`

`cd agentic-brochure-generator`

**2. Install dependencies**

`uv sync`

`uv run playwright install chromium`

**3. Configure environment variables**

Create a `.env` file containing:

`OPENAI_API_KEY=your_api_key_here`

**4. Configure the target website**

Update the `WEBSITE` configuration in `main.py`.

**5. Run the application**

`uv run python run_langgraph.py`

Generated brochures, research notes, reviews, and LinkedIn posts are saved as Markdown and JSON files in the project directory.

The first execution may download the local embedding model.

## Project Status

The core agentic workflow is functional. The project is actively being developed as a hands-on exploration of AI engineering and agent orchestration.

### Roadmap

- [x] OpenAI tool-calling Research Agent
- [x] LangChain integration
- [x] Local embeddings and semantic search
- [x] RAG and agentic retrieval
- [x] LangGraph workflow orchestration
- [ ] Persistent workflow state
- [ ] Human-in-the-loop approvals
- [ ] Observability and tracing
- [ ] Automated testing and evaluation
- [ ] API and frontend

## License

No license has been selected yet.