declare module "tabulator-tables" {
  interface TabulatorCell {
    getData(): object;
  }

  interface TabulatorColumn {
    title: string;
    field: string;
    minWidth?: number;
    width?: number;
    hozAlign?: "left" | "center" | "right";
    formatter?: (cell: TabulatorCell) => HTMLElement | string;
  }

  interface TabulatorOptions {
    height: number;
    layout: string;
    data: object[];
    placeholder: string;
    pagination: boolean;
    paginationMode: string;
    paginationSize: number;
    paginationSizeSelector: boolean;
    movableColumns: boolean;
    columns: TabulatorColumn[];
  }

  export class TabulatorFull {
    constructor(element: HTMLElement, options: TabulatorOptions);
    setData(data: object[]): Promise<void>;
    redraw(force?: boolean): void;
  }
}
