import json

from threading import Lock
from urllib.parse import urlparse

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from playwright.sync_api import sync_playwright
from sentence_transformers import SentenceTransformer

from brochure_generator.main import(
  MODEL,
  ROOT,
  WEBSITE,
  get_api_key,
  crawl_homepage,
  fetch_page_content,
  is_same_site,
)

from brochure_generator.rag_search import (
  EMBEDDING_MODEL,
  create_chunks,
  create_embeddings,
  semantic_search,
)

from brochure_generator.research_agent import SP_RESEARCH, MAX_TOOL_CALLS

# ask chatgpt to generate this
SP_AGENTIC_RESEARCH = SP_RESEARCH + """
Knowledge base workflow:
- First discover the available website links.
- Read 3 to 4 relevant pages to populate your knowledge base.
- Reading a page stores its content and returns a short preview.
- Use search_knowledge_base to retrieve detailed information.
- Search for specific topics such as services, expertise,
  industries, projects and company culture.
- You must perform at least one knowledge base search
  before producing your final research notes.
- If important information is missing, investigate another page.

Use retrieved passages as evidence for factual claims.
Include source URLs in your final research notes.
"""

def research_company_langchain(name: str, website_url: str) -> tuple[str, dict[str, str]]:
  ## keep the base code kind of the same
  hostname = urlparse(website_url).hostname
  if not hostname:
    raise ValueError("Invalid website URL")
  
  # state for the research run
  allowed_urls = {website_url}
  discovered_links: list[str] | None = None
  read_pages: dict[str, str] = {}
  tool_calls_used = 0
  
  # local knowledge base
  embedding_model: SentenceTransformer | None = None
  indexed_chunks: list[dict] = []
  indexed_embeddings = None
  indexed_page_count = 0
  
  # in order to prevent concurrent tool calls
  tool_lock = Lock()
  
  # langchain helps combining the json tool calling and the functions in one,
  # eliminating the need to build separate schemas for each tool call
  @tool
  def discover_links() -> str:
    """Discover links on the company's homepage"""
    nonlocal discovered_links, tool_calls_used
    
    with tool_lock:
      if tool_calls_used >= MAX_TOOL_CALLS:
        return "Tool limit reached."
      
      tool_calls_used += 1
      print("Tool requested: discover_links")
      
      if discovered_links is None:
        discovered_links = crawl_homepage(website_url)
        allowed_urls.update(discovered_links)
      
      return json.dumps({"links": discovered_links})
    
  @tool
  def read_page(url: str) -> str:
    """Read the webpage from discovered links."""
    nonlocal tool_calls_used
    
    with tool_lock:
      if tool_calls_used >= MAX_TOOL_CALLS:
        return "Tool limit reached."
      
      tool_calls_used += 1
      print(f"Tool requested: read_page({url})")
      
      if url not in allowed_urls:
        return "Error: URL is not in the allowed list"
      
      # reuse already collected content
      if url in read_pages:
        return read_pages[url]
      else:
        # each tool has its own playwright
        with sync_playwright() as p:
          browser = p.chromium.launch(headless=True)
          
          try:
            page = browser.new_page(locale="en-US")
            content = fetch_page_content(page, url)
            final_url = page.url
          finally:
            browser.close()
            
        if not is_same_site(final_url, hostname):
          return "Error: page went off site"
        
        if not content:
          return "Error: Could not read page."
        
        read_pages[url] = content
      
      # Print what happened (thank u chatgpt)
      print(f"  Indexed: {len(content)} characters")
        
      return json.dumps({
        "url": url,
        "status": "indexed",
        "characters": len(content),
        "preview": content[:300]
      }, ensure_ascii=False)
  
  @tool
  def search_knowledge_base(query: str) -> str:
    """Search indexed company webpages using semantic similarity."""
    nonlocal tool_calls_used
    nonlocal embedding_model
    nonlocal indexed_chunks, indexed_embeddings, indexed_page_count
    
    with tool_lock:
      if tool_calls_used >= MAX_TOOL_CALLS:
        return "Tool limit reached."
      
      tool_calls_used += 1
      print(f"Tool requested: seach_knowledge_base({query})")
      
      if not read_pages:
        return "Knowledge base is empty. Read some pages first."
      
      # Build or refresh the vector index
      if indexed_page_count != len(read_pages):
        if embedding_model is None:
          embedding_model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
          
        indexed_chunks = create_chunks(read_pages)
        
        if not indexed_chunks:
          return "No chunks available for semantic search."
        
        indexed_embeddings = create_embeddings(embedding_model, indexed_chunks)
        indexed_page_count = len(read_pages)
        print(
          f"Knowledge base indexed: "
          f"{len(read_pages)} pages, "
          f"{len(indexed_chunks)} chunks"
        )
      
      results = semantic_search(query=query, model=embedding_model, chunks=indexed_chunks, embeddings=indexed_embeddings, top_k=3)
      return json.dumps(
        {
          "query": query,
          "results": [
            {
              "url": result["url"],
              "score": round(result["score"], 3),
              "text": result["text"],
            }
            for result in results
          ]
        },
        ensure_ascii=False,
      )
  
  # The langchain agent itself
  # manages the history for you
  model = ChatOpenAI(model=MODEL, api_key=get_api_key(),)
  
  agent = create_agent(model=model, tools=[discover_links, read_page, search_knowledge_base], system_prompt=SP_AGENTIC_RESEARCH)
  result = agent.invoke({
    "messages": [
      {
        "role": "user",
        "content": (
          f"Research {name} at {website_url} "
          "for a company brochure"
        )
      }
    ]
  }, config={"recursion_limit": 24})
  
  # returns the last notes or message
  notes = result["messages"][-1].content
  
  if not isinstance(notes, str) or not notes.strip():
    raise RuntimeError("Research agent did not return notes")
  
  if not read_pages:
    raise RuntimeError("Research agent did not read pages")
  
  return notes.strip(), read_pages

def main() -> None:
  # this part of the code is almost the same
  print("Starting Research Agent (LangChain)...")
  
  research_output = ROOT / "research.md"
  research_sources = ROOT / "research_sources.md"
  
  notes, pages = research_company_langchain(WEBSITE["Name"], WEBSITE["URL"])
  research_output.write_text(notes + "\n", encoding='utf-8')
  research_sources.write_text("\n\n--\n\n".join(pages.values()) + "\n", encoding='utf-8')
  
  print(f"Research finished. Read {len(pages)} pages")
  print("Saved research.md and research_sources.md")
  
if __name__ == "__main__":
  main()