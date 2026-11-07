import * as duckdb from "@duckdb/duckdb-wasm";
import duckdbMvpWorkerUrl from "@duckdb/duckdb-wasm/dist/duckdb-browser-mvp.worker.js?url";
import duckdbMvpWasmUrl from "@duckdb/duckdb-wasm/dist/duckdb-mvp.wasm?url";

const DATASETS = [
  { fileName: "authors_enriched.parquet", tableName: "authors_enriched" },
  { fileName: "papers.parquet", tableName: "papers" },
] as const;

let connectionPromise: Promise<duckdb.AsyncDuckDBConnection> | undefined;

async function initializeConnection(): Promise<duckdb.AsyncDuckDBConnection> {
  const worker = new Worker(duckdbMvpWorkerUrl);
  const database = new duckdb.AsyncDuckDB(new duckdb.ConsoleLogger(), worker);

  await database.instantiate(duckdbMvpWasmUrl);

  for (const { fileName } of DATASETS) {
    const fileUrl = new URL(
      `${import.meta.env.BASE_URL}data/${fileName}`,
      document.baseURI,
    );
    await database.registerFileURL(
      fileName,
      fileUrl.toString(),
      duckdb.DuckDBDataProtocol.HTTP,
      false,
    );
  }

  const connection = await database.connect();
  for (const { fileName, tableName } of DATASETS) {
    await connection.query(
      `CREATE TEMP TABLE ${tableName} AS SELECT * FROM read_parquet('${fileName}')`,
    );
  }
  await connection.query(`
    CREATE TEMP TABLE author_summary AS
    SELECT
      publication_year,
      subfield,
      macro_cluster,
      country_code,
      academic_cohort,
      gender,
      author_order,
      COUNT(*) AS author_count
    FROM authors_enriched
    GROUP BY ALL
  `);
  await connection.query(`
    CREATE TEMP TABLE author_chart_summary AS
    SELECT
      publication_year,
      subfield,
      macro_cluster,
      country_code,
      academic_cohort,
      COUNT(*) AS author_count,
      COUNT(*) FILTER (WHERE gender = 'female') AS female_count,
      COUNT(*) FILTER (WHERE gender = 'male') AS male_count,
      COUNT(*) FILTER (WHERE gender = 'non_binary') AS non_binary_count,
      COUNT(*) FILTER (WHERE gender = 'unknown') AS unknown_count,
      COUNT(*) FILTER (
        WHERE author_order = 'first' AND gender = 'female'
      ) AS first_female,
      COUNT(*) FILTER (
        WHERE author_order = 'first' AND gender IN ('female', 'male')
      ) AS first_known,
      COUNT(*) FILTER (
        WHERE author_order = 'last' AND gender = 'female'
      ) AS last_female,
      COUNT(*) FILTER (
        WHERE author_order = 'last' AND gender IN ('female', 'male')
      ) AS last_known
    FROM authors_enriched
    GROUP BY
      publication_year,
      subfield,
      macro_cluster,
      country_code,
      academic_cohort
  `);
  return connection;
}

/** A shared DuckDB connection initialized on first use. */
export function getDatabaseConnection(): Promise<duckdb.AsyncDuckDBConnection> {
  if (!connectionPromise) {
    connectionPromise = initializeConnection().catch((error: unknown) => {
      connectionPromise = undefined;
      throw error;
    });
  }
  return connectionPromise;
}

/** Initializes the shared database as soon as this module is imported. */
export const databaseConnection = getDatabaseConnection();
