from pathlib import Path

import torch

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from brochure_generator.main import ROOT

SOURCE_FILE = ROOT / "research_sources.md"

# local embedding model
EMBEDDING_MODEL = (
  "sentence-transformers/"
  "paraphrase-multilingual-MiniLM-L12-v2"
)

## sets the length of each splitted text
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 3

def load_research_pages(path: Path) -> dict[str, str]:
  if not path.exists():
    raise FileNotFoundError(f"Missing research file: {path}")
  
  # just gets the full raw text
  raw = path.read_text(encoding="utf-8")
  raw = raw.replace("\n\n---\n\n", "\n\n--\n\n")
  
  pages = {}
  
  for block in raw.split("\n\n--\n\n"):
    url, separator, content = block.strip().partition("\n\n")
    
    if separator and url.startswith(("http://", "https://")):
      if content.strip():
        pages[url] = content.strip()
        
  if not pages:
    raise RuntimeError("No research pages found")
  
  return pages

def create_chunks(pages: dict[str, str]) -> list[dict]:
  ## defines the text splitter for RAG
  splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
  )
  
  chunks = []
  
  ## creates the chunks
  for url, content in pages.items():
    for chunk in splitter.split_text(content):
      chunks.append({
        "url": url,
        "text": chunk
      })
  
  return chunks

def create_embeddings(model: SentenceTransformer, chunks: list[dict]) -> torch.Tensor:
  texts = [chunk["text"] for chunk in chunks]
  
  ## running the local embedding model
  return model.encode_document(texts, convert_to_tensor=True, normalize_embeddings=True,)

def semantic_search(query: str, model: SentenceTransformer, chunks: list[dict], embeddings: torch.Tensor, top_k: int = TOP_K) -> list[dict]:
  query_vector = model.encode_query(query, convert_to_tensor=True, normalize_embeddings=True)
  
  scores = embeddings @ query_vector
  
  values, indices = torch.topk(scores, k=min(top_k, len(chunks)))
  
  results = []
  
  for score, index in zip(values.tolist(), indices.tolist()):
    results.append({
      "score": score,
      "url": chunks[index]["url"],
      "text": chunks[index]["text"],
    })
    
  return results

def main() -> None:
  pages = load_research_pages(SOURCE_FILE)
  chunks = create_chunks(pages)
  
  print(f"Loaded {len(pages)} pages")
  print(f"Created {len(chunks)} chunks")
  
  print("Loading local embedding model...")
  
  model = SentenceTransformer(
    EMBEDDING_MODEL,
    device="cpu",
  )
  
  embeddings = create_embeddings(model, chunks)
  
  print(f"Embedding dimensions: {embeddings.shape}")
  print("Semantic search ready!")
  
  queries = [
    "What services does Enginia provide?",
    "What is the company culture like?",
    "How does Enginia support manufacturing companies?"
  ]
  
  for query in queries:
    print(f"\nQUERY: {query}")
    
    results = semantic_search(query, model, chunks, embeddings)
    
    for result in results:
      print(f"\nScore: {result['score']:.3f}")
      print(f"Source: {result['url']}")
      print(result["text"][:300])
      print("---")
      
if __name__ == "__main__":
  main()