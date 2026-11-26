from openai import OpenAI
from brochure_generator.research_langchain import research_company_langchain
from sentence_transformers import SentenceTransformer

from brochure_generator.main import (
  ROOT,
  WEBSITE,
  get_api_key,
  build_brochure_input,
  create_brochure,
  run_editorial_workflow
)

from brochure_generator.rag_search import (
  EMBEDDING_MODEL,
  create_chunks,
  create_embeddings,
  semantic_search, 
)

## just ask chatgpt to generate rag queries
RAG_QUERIES = [
  "What is the company's background and expertise?",
  "What services, products and solutions does the company offer?",
  "Which industries and customers does the company serve?",
  "What projects or customer success stories are available?",
  "What is the company culture like and what careers are available?",
  "What contact information is provided?",
]

TOP_K_PER_QUERY = 2

def retrieve_brochure_context(pages: dict[str, str]) -> str:
  # 1 chunk the collected website pages
  chunks = create_chunks(pages)
  
  if not chunks:
    raise RuntimeError("No research chunks available")
  
  print(f"Created {len(chunks)} research chunks")
  
  # 2 load local embedding model and generate embeddings
  model = SentenceTransformer(
    EMBEDDING_MODEL,
    device='cpu'
  )
  
  embeddings = create_embeddings(model, chunks)
  
  # 3 retrieve the relevant chunks
  selected: list[str] = []
  seen: set[tuple[str, str]] = set()
  
  for query in RAG_QUERIES:
    print(f"Searching: {query}")
    
    results = semantic_search(query=query, model=model, chunks=chunks, embeddings=embeddings, top_k=TOP_K_PER_QUERY)
    
    for result in results:
      key = (result["url"], result["text"])
      
      # this avoids using the same chunk
      if key in seen:
        continue
      
      seen.add(key)
      selected.append(
        f"Source: {result['url']}\n\n"
        f"{result['text']}"
      )
  
  if not selected:
    raise RuntimeError("RAG returned no relevant context")
  
  print(f"Selected {len(selected)} unique chunks")
  
  return "\n\n---\n\n".join(selected)

def main() -> None:
  client = OpenAI(api_key=get_api_key())
  
  print("Starting research agent...")
  
  notes, pages = research_company_langchain(WEBSITE["Name"], WEBSITE["URL"])
  
  source_content = "\n\n--\n\n".join(pages.values())
  
  research_output = ROOT / "research.md"
  research_sources = ROOT / "research_sources.md"
  brochure_path = ROOT / "brochure_agentic.md"
  review_path = ROOT / "review_agentic.json"
  linkedin_path = ROOT / "linkedin_agentic.md"
  rag_path = ROOT / "rag_context.md"
  
  research_output.write_text(notes + "\n", encoding='utf-8')
  research_sources.write_text(source_content + "\n", encoding='utf-8')
  
  print(f"Research completed. Read {len(pages)} pages.")
  print("Generating initial brochure...")
  
  print("Building RAG context...")
  rag_context = retrieve_brochure_context(pages)
  
  # save the retrieved context
  rag_path.write_text(rag_context + "\n", encoding='utf-8')
  
  print(
    f"Original content: {len(source_content)} characters\n"
    f"Retrieved context: {len(rag_context)} characters"
  )
  
  brochure_input = build_brochure_input(WEBSITE["Name"], WEBSITE["URL"], rag_context,)
  initial_brocure = create_brochure(client, brochure_input)
  
  print("Starting editorial workflow...")
  brochure, review, social_post = run_editorial_workflow(client, rag_context, initial_brocure)
  brochure_path.write_text(brochure + "\n", encoding='utf-8')
  review_path.write_text(review.model_dump_json(indent=2), encoding="utf-8")
  
  if social_post is not None:
    hashtags = " ".join(f"#{tag.strip().lstrip('#').replace(' ', '')}" for tag in social_post.hashtags if tag.strip().lstrip("#"))
    linkedin = (
      f"# {social_post.title}\n\n"
      f"{social_post.content}\n\n"
      f"{hashtags}\n"
    )
    linkedin_path.write_text(linkedin, encoding='utf-8')
  
  print("Agentic workflow completed!")
  
if __name__ == "__main__":
  main()