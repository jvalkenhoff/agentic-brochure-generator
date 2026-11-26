from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from brochure_generator.main import (
  ROOT,
  WEBSITE,
  MAX_REVISIONS,
  BrochureReview,
  SocialPost,
  build_brochure_input,
  create_brochure,
  create_linkedin_post,
  get_api_key,
  review_brochure,
  revise_brochure
)

from brochure_generator.research_langchain import research_company_langchain
from brochure_generator.run_agentic import retrieve_brochure_context

class BrochureState(TypedDict, total=False):
  name: str
  website_url: str
  notes: str
  pages: dict[str, str]
  rag_context: str
  brochure: str
  review: BrochureReview
  revision_count: int
  social_post: SocialPost
  
def build_workflow(client: OpenAI):
  # each node receives the current shared state and returns ONLY its updates
  def research(state: BrochureState) -> dict:
    print("[research] Gathering website information...")
    notes, pages = research_company_langchain(state["name"], state["website_url"])
    return { "notes": notes, "pages": pages}
  
  def retrieve(state: BrochureState) -> dict:
    print("[retrieve] Building RAG context...")
    context = retrieve_brochure_context(state["pages"])
    return {"rag_context": context}
  
  def write(state: BrochureState) -> dict:
    print("[writer] Generating brochure...")
    prompt = build_brochure_input(state['name'], state['website_url'], state['rag_context'])
    brochure = create_brochure(client, prompt)
    return {"brochure": brochure}
    
  def review(state: BrochureState) -> dict:
    print("[reviewer] Evaluating brochure...")
    result = review_brochure(client, state['rag_context'], state['brochure'])
    print(f"[reviewer] Score: {result.score}/10 | Approved: {result.approved}")
    return {"review": result}
  
  def revise(state: BrochureState) -> dict:
    number = state["revision_count"] + 1
    print(f"[writer] Revision {number}/{MAX_REVISIONS}...")
    revised = revise_brochure(client, state['rag_context'], state['brochure'], state['review'])
    return {"brochure": revised, "revision_count": number}
  
  def social(state: BrochureState) -> dict:
    print("[social] Generating LinkedIn post...")
    post = create_linkedin_post(client, state["brochure"])
    return {"social_post": post}
  
  def route_review(state: BrochureState) -> Literal["approved", "revise", "stop"]:
    if state["review"].approved:
        return "approved"

    if state["revision_count"] < MAX_REVISIONS:
        return "revise"

    return "stop"
  
  graph = StateGraph(BrochureState)
  
  graph.add_node("research", research)
  graph.add_node("retrieve", retrieve)
  graph.add_node("write", write)
  graph.add_node("review", review)
  graph.add_node("revise", revise)
  graph.add_node("social", social)
  
  graph.add_edge(START, "research")
  graph.add_edge("research", "retrieve")
  graph.add_edge("retrieve", "write")
  graph.add_edge("write", "review")
  graph.add_conditional_edges(
    "review",
    route_review,
    {"approved": "social", "revise": "revise", "stop": END}
  )
  graph.add_edge("revise", "review")
  graph.add_edge("social", END)
  
  return graph.compile()

def main() -> None:
  client = OpenAI(api_key=get_api_key())
  
  workflow_output = ROOT / "workflow.mmd"
  research_output = ROOT / "research.md"
  research_sources = ROOT / "research_sources.md"
  brochure_path = ROOT / "brochure_agentic.md"
  review_path = ROOT / "review_agentic.json"
  linkedin_path = ROOT / "linkedin_agentic.md"
  rag_path = ROOT / "rag_context.md"
  langgraph_review = ROOT / "review_langgraph.json"
  workflow = build_workflow(client)
  # save the graph
  workflow_output.write_text(workflow.get_graph().draw_mermaid(), encoding='utf-8')  
  
  result = workflow.invoke({
    "name": WEBSITE["Name"],
    "website_url": WEBSITE["URL"],
    "revision_count": 0
  }, config={"recursion_limit": 30})
  
  research_output.write_text(result["notes"] + "\n", encoding='utf-8')
  research_sources.write_text("\n\n--\n\n".join(result["pages"].values()) + "\n", encoding="utf-8")
  rag_path.write_text(result["rag_context"] + "\n", encoding="utf-8")
  brochure_path.write_text(result["brochure"] + "\n", encoding="utf-8")
  langgraph_review.write_text(result["review"].model_dump_json(indent=2), encoding="utf-8")
  
  if result["review"].approved:
    post = result["social_post"]
    hashtags = " ".join(
      f"#{tag.strip().lstrip('#').replace(' ', '')}" for tag in post.hashtags if tag.strip().lstrip("#")
    )
    linkedin = f"# {post.title}\n\n{post.content}\n\n{hashtags}\n"
    linkedin_path.write_text(linkedin, encoding="utf-8")
    print("[done] Approved brochure and LinkedIn post saved.")
  else:
    print("[done] Review limit reached; draft saved for manual review.")
  
if __name__ == "__main__":
  main()