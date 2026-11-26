
import pytest

from brochure_generator import run_langgraph as workflow_module
from brochure_generator.main import BrochureReview, SocialPost


@pytest.mark.parametrize(
    ("decisions", "expected_revisions", "expect_social"),
    [
        pytest.param(
            [True], 0, True,
            id="approved_immediately",
        ),
        pytest.param(
            [False, True], 1, True,
            id="approved_after_revision",
        ),
        pytest.param(
            [False, False, False], 2, False,
            id="revision_limit_reached",
        ),
    ],
)
def test_workflow_routing(
    monkeypatch,
    decisions,
    expected_revisions,
    expect_social,
):
    # Simulate the reviewer's decisions
    review_decisions = iter(decisions)
    review_count = 0

    # Fake the Research Agent
    def fake_research(name, url):
        return (
            "Research notes",
            {"https://example.com": "Company information"},
        )

    # Fake semantic retrieval
    def fake_retrieve(pages):
        return "Relevant company information"

    # Fake brochure generation
    def fake_writer(client, prompt):
        return "Initial brochure"

    # Fake editorial review
    def fake_reviewer(client, source, brochure):
        nonlocal review_count

        review_count += 1
        approved = next(review_decisions)

        return BrochureReview(
            approved=approved,
            score=9 if approved else 5,
            strengths=[],
            weaknesses=[],
            suggestions=[],
        )

    # Fake the revision process
    def fake_revision(client, source, brochure, review):
        return brochure + " (revised)"

    # Fake social media generation
    def fake_social(client, brochure):
        return SocialPost(
            title="Company overview",
            content="Generated LinkedIn content",
            hashtags=["engineering"],
        )

    # Replace external operations with our fake functions
    monkeypatch.setattr(
        workflow_module,
        "research_company_langchain",
        fake_research,
    )
    monkeypatch.setattr(
        workflow_module,
        "retrieve_brochure_context",
        fake_retrieve,
    )
    monkeypatch.setattr(
        workflow_module,
        "create_brochure",
        fake_writer,
    )
    monkeypatch.setattr(
        workflow_module,
        "review_brochure",
        fake_reviewer,
    )
    monkeypatch.setattr(
        workflow_module,
        "revise_brochure",
        fake_revision,
    )
    monkeypatch.setattr(
        workflow_module,
        "create_linkedin_post",
        fake_social,
    )

    # Execute the real LangGraph workflow
    graph = workflow_module.build_workflow(client=object())

    result = graph.invoke(
        {
            "name": "Example Company",
            "website_url": "https://example.com",
            "revision_count": 0,
        },
        config={"recursion_limit": 30},
    )

    # Verify graph behavior
    assert result["revision_count"] == expected_revisions
    assert result["review"].approved == expect_social
    assert review_count == len(decisions)
    assert ("social_post" in result) == expect_social

    assert (
        result["brochure"].count("(revised)")
        == expected_revisions
    )
