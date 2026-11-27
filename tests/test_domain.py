
import pytest

from pydantic import ValidationError

from brochure_generator.domain.company import CompanyProfile
from brochure_generator.domain.content_brief import ContentBrief


def test_company_profile_valid():
    company = CompanyProfile(
        id="enginia",
        name="Enginia",
        website_url="https://enginia.nl/",
    )

    assert company.id == "enginia"
    assert company.name == "Enginia"
    assert company.brand_voice == ""


@pytest.mark.parametrize(
    "invalid_id",
    ["Enginia", "invalid company", ""],
)
def test_company_rejects_invalid_id(invalid_id):
    with pytest.raises(ValidationError):
        CompanyProfile(
            id=invalid_id,
            name="Enginia",
            website_url="https://enginia.nl/",
        )


def test_company_rejects_invalid_website():
    with pytest.raises(ValidationError):
        CompanyProfile(
            id="enginia",
            name="Enginia",
            website_url="ftp://enginia.nl/",
        )


def test_company_rejects_empty_name():
    with pytest.raises(ValidationError):
        CompanyProfile(
            id="enginia",
            name="   ",
            website_url="https://enginia.nl/",
        )


def test_content_brief_valid():
    brief = ContentBrief(
        company_id="enginia",
        instructions="Write a company brochure",
        target_audience="Engineering companies",
    )

    assert brief.content_type == "brochure"
    assert brief.company_id == "enginia"
    assert brief.tone_override is None


def test_content_brief_rejects_empty_instructions():
    with pytest.raises(ValidationError):
        ContentBrief(
            company_id="enginia",
            instructions="   ",
        )


def test_content_brief_rejects_unsupported_type():
    with pytest.raises(ValidationError):
        ContentBrief(
            company_id="enginia",
            content_type="podcast",
            instructions="Create a podcast script",
        )
