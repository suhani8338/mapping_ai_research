import { getDatabaseConnection } from "./duckdb";

interface QueryTable {
  readonly numRows: number;
  readonly schema: {
    readonly fields: readonly { readonly name: string }[];
  };
  getChild(name: string): { get(index: number): unknown } | null;
}

export interface FilterState {
  macroClusters: string[];
  subfields: string[];
  countries: string[];
  academicCohorts: string[];
  genders: string[];
  yearRange: [number, number];
}

export interface MetricRow {
  subfield: string;
  macro_cluster: string;
  total_authors: number;
  known_authors: number;
  female_count: number;
  female_pct: number;
  female_total_pct: number;
  male_pct: number;
  non_binary_pct: number;
  unknown_pct: number;
}

export interface CountryMetricRow {
  country_code: string;
  total_authors: number;
  known_authors: number;
  female_pct: number;
}

export interface PaperBrowserRow {
  paper_title: string;
  subfield: string;
  publication_year: number;
  first_author: string;
  senior_author: string;
  country_code: string;
  arxiv_url: string;
}

export interface FilterOptions {
  macroClusters: string[];
  subfields: string[];
  countries: string[];
  academicCohorts: string[];
  genders: string[];
  yearRange: [number, number];
}

export interface PipelineDropoffRow {
  subfield: string;
  p_first: number | null;
  p_last: number | null;
  delta: number | null;
  first_known_authors: number;
  last_known_authors: number;
}

const AUTHORS_TABLE = "authors_enriched";
const PAPERS_TABLE = "papers";
const AUTHOR_SUMMARY_TABLE = "author_summary";
const AUTHOR_CHART_SUMMARY_TABLE = "author_chart_summary";
const MIN_PUBLICATION_YEAR = 2020;

function sqlString(value: string): string {
  return `'${value.replaceAll("'", "''")}'`;
}

function listFilter(column: string, values: string[]): string | undefined {
  if (values.length === 0) {
    return undefined;
  }
  return `${column} IN (${values.map(sqlString).join(", ")})`;
}

function buildAuthorWhereClause(
  filters: FilterState,
  tableAlias?: string,
  includeGender = true,
): string {
  const column = (name: string) => (tableAlias ? `${tableAlias}.${name}` : name);
  const clauses = [
    listFilter(column("macro_cluster"), filters.macroClusters),
    listFilter(column("subfield"), filters.subfields),
    listFilter(column("country_code"), filters.countries),
    listFilter(column("academic_cohort"), filters.academicCohorts),
    includeGender ? listFilter(column("gender"), filters.genders) : undefined,
  ].filter((clause): clause is string => clause !== undefined);
  const [minimumYear, maximumYear] = filters.yearRange;

  if (
    !Number.isSafeInteger(minimumYear) ||
    !Number.isSafeInteger(maximumYear) ||
    minimumYear < MIN_PUBLICATION_YEAR ||
    minimumYear > maximumYear
  ) {
    throw new RangeError(
      `yearRange must contain ordered safe integer years from ${MIN_PUBLICATION_YEAR} onward`,
    );
  }

  clauses.push(
    `publication_year BETWEEN ${minimumYear} AND ${maximumYear}`,
  );
  return clauses.join(" AND ");
}

function tableValue(
  table: QueryTable,
  columnName: string,
  rowIndex: number,
): unknown {
  const column = table.getChild(columnName);
  if (!column) {
    throw new Error(`DuckDB query result is missing column "${columnName}"`);
  }
  return column.get(rowIndex);
}

function numericValue(
  table: QueryTable,
  columnName: string,
  rowIndex: number,
): number {
  const value = tableValue(table, columnName, rowIndex);
  return value === null || value === undefined ? 0 : Number(value);
}

function nullableNumericValue(
  table: QueryTable,
  columnName: string,
  rowIndex: number,
): number | null {
  const value = tableValue(table, columnName, rowIndex);
  return value === null || value === undefined ? null : Number(value);
}

function percentage(numerator: number, denominator: number): number {
  return denominator === 0 ? 0 : (100 * numerator) / denominator;
}

function stringValue(
  table: QueryTable,
  columnName: string,
  rowIndex: number,
): string {
  const value = tableValue(table, columnName, rowIndex);
  return value === null || value === undefined ? "" : String(value);
}

async function distinctValues(columnName: string): Promise<string[]> {
  const connection = await getDatabaseConnection();
  const result = await connection.query(`
    SELECT DISTINCT ${columnName} AS value
    FROM ${AUTHOR_SUMMARY_TABLE}
    WHERE ${columnName} IS NOT NULL AND ${columnName} <> ''
    ORDER BY value
  `);
  return Array.from({ length: result.numRows }, (_, rowIndex) =>
    stringValue(result, "value", rowIndex),
  );
}

export async function queryFilterOptions(): Promise<FilterOptions> {
  const connection = await getDatabaseConnection();
  const years = await connection.query(`
    SELECT MIN(publication_year) AS minimum_year,
           MAX(publication_year) AS maximum_year
    FROM ${AUTHOR_SUMMARY_TABLE}
    WHERE publication_year >= ${MIN_PUBLICATION_YEAR}
  `);
  const minimumYear = nullableNumericValue(years, "minimum_year", 0);
  const maximumYear = nullableNumericValue(years, "maximum_year", 0);
  if (minimumYear === null || maximumYear === null) {
    throw new Error(
      `No author records are available from ${MIN_PUBLICATION_YEAR} onward`,
    );
  }
  const [macroClusters, subfields, countries, academicCohorts, genders] =
    await Promise.all([
      distinctValues("macro_cluster"),
      distinctValues("subfield"),
      distinctValues("country_code"),
      distinctValues("academic_cohort"),
      distinctValues("gender"),
    ]);

  return {
    macroClusters,
    subfields,
    countries,
    academicCohorts,
    genders,
    yearRange: [Math.max(MIN_PUBLICATION_YEAR, minimumYear), maximumYear],
  };
}

export async function querySubfieldMetrics(
  filters: FilterState,
): Promise<MetricRow[]> {
  const whereClause = buildAuthorWhereClause(filters, undefined, false);
  const connection = await getDatabaseConnection();
  const result = await connection.query(`
    SELECT
      subfield,
      macro_cluster,
      SUM(author_count) AS total_authors,
      SUM(female_count) AS female_count,
      SUM(male_count) AS male_count,
      SUM(non_binary_count) AS non_binary_count,
      SUM(unknown_count) AS unknown_count
    FROM ${AUTHOR_CHART_SUMMARY_TABLE}
    WHERE ${whereClause}
    GROUP BY subfield, macro_cluster
    ORDER BY subfield
  `);

  return Array.from({ length: result.numRows }, (_, rowIndex) => {
    const totalAuthors = numericValue(result, "total_authors", rowIndex);
    const femaleCount = numericValue(result, "female_count", rowIndex);
    const maleCount = numericValue(result, "male_count", rowIndex);
    return {
      subfield: stringValue(result, "subfield", rowIndex),
      macro_cluster: stringValue(result, "macro_cluster", rowIndex),
      total_authors: totalAuthors,
      known_authors: femaleCount + maleCount,
      female_count: femaleCount,
      female_pct: percentage(femaleCount, femaleCount + maleCount),
      female_total_pct: percentage(femaleCount, totalAuthors),
      male_pct: percentage(maleCount, totalAuthors),
      non_binary_pct: percentage(
        numericValue(result, "non_binary_count", rowIndex),
        totalAuthors,
      ),
      unknown_pct: percentage(
        numericValue(result, "unknown_count", rowIndex),
        totalAuthors,
      ),
    };
  });
}

export async function queryPipelineDropoff(
  filters: FilterState,
): Promise<PipelineDropoffRow[]> {
  const whereClause = buildAuthorWhereClause(filters, undefined, false);
  const connection = await getDatabaseConnection();
  const result = await connection.query(`
    WITH role_metrics AS (
      SELECT
        subfield,
        SUM(first_known) AS first_known_authors,
        SUM(last_known) AS last_known_authors,
        SUM(first_female) AS first_female,
        SUM(last_female) AS last_female
      FROM ${AUTHOR_CHART_SUMMARY_TABLE}
      WHERE ${whereClause}
      GROUP BY subfield
    )
    SELECT
      subfield,
      first_female * 1.0 / NULLIF(first_known_authors, 0) AS p_first,
      last_female * 1.0 / NULLIF(last_known_authors, 0) AS p_last,
      first_female * 1.0 / NULLIF(first_known_authors, 0)
        - last_female * 1.0 / NULLIF(last_known_authors, 0) AS delta,
      first_known_authors,
      last_known_authors
    FROM role_metrics
    ORDER BY subfield
  `);

  return Array.from({ length: result.numRows }, (_, rowIndex) => ({
    subfield: stringValue(result, "subfield", rowIndex),
    p_first: nullableNumericValue(result, "p_first", rowIndex),
    p_last: nullableNumericValue(result, "p_last", rowIndex),
    delta: nullableNumericValue(result, "delta", rowIndex),
    first_known_authors: numericValue(result, "first_known_authors", rowIndex),
    last_known_authors: numericValue(result, "last_known_authors", rowIndex),
  }));
}

export async function queryCountryMetrics(
  filters: FilterState,
): Promise<CountryMetricRow[]> {
  const whereClause = buildAuthorWhereClause(filters, undefined, false);
  const connection = await getDatabaseConnection();
  const result = await connection.query(`
    SELECT
      country_code,
      SUM(author_count) AS total_authors,
      SUM(female_count) AS female_count,
      SUM(male_count) AS male_count
    FROM ${AUTHOR_CHART_SUMMARY_TABLE}
    WHERE ${whereClause}
      AND country_code <> 'UNKNOWN'
      AND length(country_code) = 2
    GROUP BY country_code
    ORDER BY total_authors DESC, country_code
  `);

  return Array.from({ length: result.numRows }, (_, rowIndex) => {
    const femaleCount = numericValue(result, "female_count", rowIndex);
    const maleCount = numericValue(result, "male_count", rowIndex);
    return {
      country_code: stringValue(result, "country_code", rowIndex),
      total_authors: numericValue(result, "total_authors", rowIndex),
      known_authors: femaleCount + maleCount,
      female_pct: percentage(femaleCount, femaleCount + maleCount),
    };
  });
}

export async function queryPaperBrowserRows(
  filters: FilterState,
): Promise<PaperBrowserRow[]> {
  const whereClause = buildAuthorWhereClause(filters, "authors");
  const connection = await getDatabaseConnection();
  const result = await connection.query(`
    WITH selected_authors AS (
      SELECT
        authors.paper_id,
        authors.subfield,
        authors.publication_year,
        authors.full_name,
        authors.author_order,
        authors.country_code
      FROM ${AUTHORS_TABLE} AS authors
      WHERE ${whereClause}
    )
    SELECT
      papers.title AS paper_title,
      papers.subfield,
      selected_authors.publication_year,
      MAX(selected_authors.full_name)
        FILTER (WHERE selected_authors.author_order IN ('first', 'sole')) AS first_author,
      MAX(selected_authors.full_name)
        FILTER (WHERE selected_authors.author_order IN ('last', 'sole')) AS senior_author,
      MAX(selected_authors.country_code)
        FILTER (WHERE selected_authors.author_order IN ('last', 'sole')) AS country_code,
      papers.arxiv_url
    FROM selected_authors
    INNER JOIN ${PAPERS_TABLE} AS papers
      ON selected_authors.paper_id = papers.paper_id
      AND selected_authors.subfield = papers.subfield
    GROUP BY
      selected_authors.paper_id,
      papers.title,
      papers.subfield,
      selected_authors.publication_year,
      papers.arxiv_url
    ORDER BY selected_authors.publication_year DESC, papers.title
  `);

  return Array.from({ length: result.numRows }, (_, rowIndex) => ({
    paper_title: stringValue(result, "paper_title", rowIndex),
    subfield: stringValue(result, "subfield", rowIndex),
    publication_year: numericValue(result, "publication_year", rowIndex),
    first_author: stringValue(result, "first_author", rowIndex),
    senior_author: stringValue(result, "senior_author", rowIndex),
    country_code: stringValue(result, "country_code", rowIndex),
    arxiv_url: stringValue(result, "arxiv_url", rowIndex),
  }));
}

function csvCell(value: unknown): string {
  if (value === null || value === undefined) {
    return "";
  }
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

export async function exportFilteredDataCSV(
  filters: FilterState,
): Promise<Blob> {
  const whereClause = buildAuthorWhereClause(filters, "authors");
  const connection = await getDatabaseConnection();
  const result = await connection.query(`
    SELECT
      authors.*,
      papers.title AS paper_title,
      papers.primary_category,
      papers.publication_date,
      papers.arxiv_url,
      papers.total_authors AS paper_total_authors
    FROM ${AUTHORS_TABLE} AS authors
    INNER JOIN ${PAPERS_TABLE} AS papers
      ON authors.paper_id = papers.paper_id
      AND authors.subfield = papers.subfield
    WHERE ${whereClause}
    ORDER BY authors.paper_id, authors.author_order, authors.full_name
  `);

  const headers = result.schema.fields.map((field) => field.name);
  const rows = Array.from({ length: result.numRows }, (_, rowIndex) =>
    result.schema.fields
      .map((field) => csvCell(tableValue(result, field.name, rowIndex)))
      .join(","),
  );
  const csv = [headers.map(csvCell).join(","), ...rows].join("\r\n");
  return new Blob([csv], { type: "text/csv;charset=utf-8" });
}
