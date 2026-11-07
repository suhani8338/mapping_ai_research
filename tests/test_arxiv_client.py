from src.ingestion.arxiv_client import ArxivClient


ATOM_FEED = """\
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>https://arxiv.org/abs/2501.00001v1</id>
    <title>Example paper</title>
    <published>2025-01-01T00:00:00Z</published>
    <author><name>First Author</name></author>
    <author><name>Middle Author</name></author>
    <author><name>Last Author</name></author>
    <category term="cs.CL" />
    <arxiv:primary_category term="cs.CL" />
  </entry>
</feed>
"""


class FakeResponse:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self):
        self.responses = [FakeResponse(503), FakeResponse(200, ATOM_FEED)]
        self.calls = 0
        self.urls = []

    def get(self, url, *, timeout):
        self.calls += 1
        self.urls.append(url)
        return self.responses.pop(0)


def test_retries_arxiv_service_unavailable(monkeypatch):
    sleeps = []
    monkeypatch.setattr("src.ingestion.arxiv_client.time.sleep", sleeps.append)
    session = FakeSession()
    client = ArxivClient(session=session, delay_seconds=0)

    entries = client.query("cat:cs.CL")

    assert session.calls == 2
    assert sleeps == [1]
    assert entries[0]["paper_id"] == "2501.00001"
    assert entries[0]["published"] == "2025-01-01T00:00:00Z"
    assert entries[0]["primary_category"] == "cs.CL"
    assert entries[0]["authors"] == ["First Author", "Middle Author", "Last Author"]
    assert [author["author_order"] for author in entries[0]["author_records"]] == [
        "first",
        "middle",
        "last",
    ]
    assert "sortBy=relevance" in session.urls[-1]


def test_single_author_is_tagged_as_first():
    feed = ATOM_FEED.replace(
        "<author><name>Middle Author</name></author>\n"
        "    <author><name>Last Author</name></author>",
        "",
    )

    entry = ArxivClient._parse_feed(feed)[0]

    assert entry["author_records"] == [
        {"full_name": "First Author", "author_order": "first"}
    ]


def test_author_email_is_retained_when_present_in_atom_metadata():
    feed = ATOM_FEED.replace(
        "<author><name>First Author</name></author>",
        "<author><name>First Author</name>"
        "<arxiv:email>first@example.jp</arxiv:email></author>",
    )

    entry = ArxivClient._parse_feed(feed)[0]

    assert entry["author_records"][0]["email"] == "first@example.jp"


def test_paper_doi_is_retained_when_present_in_atom_metadata():
    feed = ATOM_FEED.replace(
        "<title>Example paper</title>",
        "<title>Example paper</title><arxiv:doi>10.1234/example</arxiv:doi>",
    )

    entry = ArxivClient._parse_feed(feed)[0]

    assert entry["doi"] == "10.1234/example"


def test_query_supports_ascending_submitted_date_order():
    session = FakeSession()
    session.responses = [FakeResponse(200, ATOM_FEED)]
    client = ArxivClient(session=session, delay_seconds=0)

    client.query(
        'au:"First Author"',
        max_results=100,
        sort_by="submittedDate",
        sort_order="ascending",
    )

    assert "sortBy=submittedDate" in session.urls[0]
    assert "sortOrder=ascending" in session.urls[0]
