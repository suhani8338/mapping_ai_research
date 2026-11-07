"""Arrow schemas for the static application's Parquet datasets."""

import pyarrow as pa

AUTHORS_ENRICHED_SCHEMA = pa.schema(
    [
        pa.field("paper_id", pa.string()),
        pa.field("subfield", pa.dictionary(pa.int16(), pa.string())),
        pa.field("macro_cluster", pa.string()),
        pa.field("publication_year", pa.int32()),
        pa.field("author_order", pa.string()),
        pa.field("full_name", pa.string()),
        pa.field("gender", pa.string()),
        pa.field("gender_confidence", pa.float32()),
        pa.field("country_code", pa.dictionary(pa.int16(), pa.string())),
        pa.field("academic_age", pa.int16()),
        pa.field("academic_cohort", pa.string()),
    ]
)

PAPERS_SCHEMA = pa.schema(
    [
        pa.field("paper_id", pa.string()),
        pa.field("title", pa.string()),
        pa.field("primary_category", pa.string()),
        pa.field("subfield", pa.dictionary(pa.int16(), pa.string())),
        pa.field("macro_cluster", pa.string()),
        pa.field("publication_date", pa.string()),
        pa.field("arxiv_url", pa.string()),
        pa.field("total_authors", pa.int16()),
    ]
)

GENDER_LABELS = frozenset({"female", "male", "non_binary", "unknown"})
