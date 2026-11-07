import { TabulatorFull as Tabulator } from "tabulator-tables";
import type { PaperBrowserRow } from "../db/queries";

export class DataTable {
  private readonly table: Tabulator;

  constructor(
    selector: string,
    onExport: () => Promise<void>,
    onError: (error: unknown) => void,
  ) {
    const tableElement = document.querySelector<HTMLElement>(selector);
    const exportButton = document.querySelector<HTMLButtonElement>("#download-csv");
    if (!tableElement || !exportButton) {
      throw new Error("Missing paper table or CSV export button");
    }

    this.table = new Tabulator(tableElement, {
      height: 560,
      layout: "fitColumns",
      data: [],
      placeholder: "No papers match these filters.",
      pagination: true,
      paginationMode: "local",
      paginationSize: 50,
      paginationSizeSelector: false,
      movableColumns: true,
      columns: [
        {
          title: "Paper title",
          field: "paper_title",
          minWidth: 280,
          formatter: (cell) => {
            const row = cell.getData() as PaperBrowserRow;
            const link = document.createElement("a");
            link.href = row.arxiv_url;
            link.target = "_blank";
            link.rel = "noopener noreferrer";
            link.textContent = row.paper_title;
            return link;
          },
        },
        { title: "Subfield", field: "subfield", minWidth: 190 },
        { title: "Year", field: "publication_year", width: 90, hozAlign: "right" },
        { title: "First author", field: "first_author", minWidth: 150 },
        { title: "Senior author", field: "senior_author", minWidth: 150 },
        { title: "Country", field: "country_code", width: 100 },
      ],
    });

    exportButton.addEventListener("click", async () => {
      exportButton.disabled = true;
      try {
        await onExport();
      } catch (error) {
        onError(error);
      } finally {
        exportButton.disabled = false;
      }
    });
  }

  async setData(rows: PaperBrowserRow[]): Promise<void> {
    await this.table.setData(rows);
  }

  redraw(): void {
    this.table.redraw(true);
  }
}
