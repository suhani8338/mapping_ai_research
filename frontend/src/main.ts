import "./style.css";
import { databaseConnection } from "./db/duckdb";
import {
  exportFilteredDataCSV,
  queryCountryMetrics,
  queryFilterOptions,
  queryPaperBrowserRows,
  queryPipelineDropoff,
  querySubfieldMetrics,
  type FilterOptions,
  type FilterState,
} from "./db/queries";
import { Charts } from "./components/Charts";
import { DataTable } from "./components/DataTable";
import { FilterSidebar } from "./components/FilterSidebar";

const status = document.querySelector<HTMLElement>("#database-status");
const filterContainer = document.querySelector<HTMLElement>("#filter-sidebar");
const charts = new Charts();
const table = new DataTable("#paper-table", async () => {
  if (!filters) return;
  const csv = await exportFilteredDataCSV(filters);
  const url = URL.createObjectURL(csv);
  const link = document.createElement("a");
  link.href = url;
  link.download = "gendered-ai-research-filtered.csv";
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  status!.textContent = "Filtered CSV download started.";
}, showError);

let filters: FilterState | undefined;
let refreshSequence = 0;

if (!status || !filterContainer) {
  throw new Error("Missing required dashboard elements");
}

function showError(error: unknown): void {
  console.error("Could not update the dashboard.", error);
  status!.textContent = "Could not update the dashboard. See the console for details.";
  status!.classList.add("status-error");
}

async function refreshDashboard(nextFilters: FilterState): Promise<void> {
  filters = nextFilters;
  const sequence = ++refreshSequence;
  status!.textContent = "Updating results…";
  status!.classList.remove("status-error");

  try {
    const [metrics, dropoff, countries, paperRows] = await Promise.all([
      querySubfieldMetrics(nextFilters),
      queryPipelineDropoff(nextFilters),
      queryCountryMetrics(nextFilters),
      queryPaperBrowserRows(nextFilters),
    ]);
    if (sequence !== refreshSequence) return;
    charts.render(metrics, dropoff, countries);
    await table.setData(paperRows);
    status!.textContent = `${paperRows.length.toLocaleString()} papers in the current selection.`;
  } catch (error) {
    showError(error);
  }
}

async function initialize(): Promise<void> {
  try {
    await databaseConnection;
    const options: FilterOptions = await queryFilterOptions();
    const sidebar = new FilterSidebar(filterContainer!, options, refreshDashboard);
    filters = sidebar.getState();
    await refreshDashboard(filters);
  } catch (error) {
    showError(error);
  }
}

for (const tab of document.querySelectorAll<HTMLButtonElement>("[data-tab]")) {
  tab.addEventListener("click", () => {
    const name = tab.dataset.tab;
    for (const button of document.querySelectorAll<HTMLButtonElement>("[data-tab]")) {
      button.setAttribute("aria-selected", String(button === tab));
    }
    for (const panel of document.querySelectorAll<HTMLElement>("[data-panel]")) {
      panel.hidden = panel.dataset.panel !== name;
    }
    if (name === "papers") table.redraw();
    else charts.resize();
  });
}

void initialize();
