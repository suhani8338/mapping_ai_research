import type { FilterOptions, FilterState } from "../db/queries";

type FilterKey = "macroClusters" | "subfields" | "countries" | "academicCohorts" | "genders";

const EMPTY_SELECTION = "__no_matching_value__";

function displayLabel(value: string): string {
  const acronyms: Record<string, string> = { ai: "AI", nlp: "NLP", edtech: "EdTech" };
  return value.split("_").map((word) =>
    acronyms[word] ?? `${word.charAt(0).toUpperCase()}${word.slice(1)}`,
  ).join(" ");
}

export class FilterSidebar {
  private readonly state: FilterState;

  constructor(
    private readonly container: HTMLElement,
    private readonly options: FilterOptions,
    private readonly onChange: (filters: FilterState) => void | Promise<void>,
  ) {
    this.state = {
      macroClusters: [],
      subfields: [],
      countries: [],
      academicCohorts: [],
      genders: [],
      yearRange: [...options.yearRange],
    };
    this.render();
  }

  getState(): FilterState {
    return {
      ...this.state,
      macroClusters: [...this.state.macroClusters],
      subfields: [...this.state.subfields],
      countries: [...this.state.countries],
      academicCohorts: [...this.state.academicCohorts],
      genders: [...this.state.genders],
      yearRange: [...this.state.yearRange],
    };
  }

  private render(): void {
    this.container.replaceChildren();
    this.addChoiceGroup("Macro cluster", "macroClusters", this.options.macroClusters);
    this.addChoiceGroup("Subfield", "subfields", this.options.subfields);
    this.addChoiceGroup("Country of affiliation", "countries", this.options.countries);
    this.addChoiceGroup("Career cohort", "academicCohorts", this.options.academicCohorts);
    this.addChoiceGroup("Gender (paper browser and export)", "genders", this.options.genders);
    this.addYearRange();

    const reset = document.createElement("button");
    reset.className = "button button-secondary reset-button";
    reset.type = "button";
    reset.textContent = "Reset filters";
    reset.addEventListener("click", () => {
      this.state.macroClusters = [];
      this.state.subfields = [];
      this.state.countries = [];
      this.state.academicCohorts = [];
      this.state.genders = [];
      this.state.yearRange = [...this.options.yearRange];
      this.render();
      void this.onChange(this.getState());
    });
    this.container.append(reset);
  }

  private addChoiceGroup(title: string, key: FilterKey, values: string[]): void {
    const fieldset = document.createElement("fieldset");
    fieldset.className = "filter-group";
    const legend = document.createElement("legend");
    legend.textContent = title;
    fieldset.append(legend);

    if (values.length === 0) {
      const empty = document.createElement("p");
      empty.className = "filter-empty";
      empty.textContent = "No values available";
      fieldset.append(empty);
    }

    const selection = this.state[key];
    const allSelected = selection.length === 0;
    for (const value of values) {
      const label = document.createElement("label");
      label.className = "filter-option";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = value;
      checkbox.checked = allSelected || selection.includes(value);
      checkbox.addEventListener("change", () => this.updateSelection(key, values, checkbox));
      const text = document.createElement("span");
      text.textContent = displayLabel(value);
      label.append(checkbox, text);
      fieldset.append(label);
    }
    this.container.append(fieldset);
  }

  private updateSelection(key: FilterKey, values: string[], changed: HTMLInputElement): void {
    const current = this.state[key].length === 0 ? [...values] : this.state[key].filter(
      (value) => value !== EMPTY_SELECTION,
    );
    const selected = changed.checked
      ? [...current, changed.value]
      : current.filter((value) => value !== changed.value);
    const uniqueSelected = [...new Set(selected)];
    this.state[key] = uniqueSelected.length === 0
      ? [EMPTY_SELECTION]
      : uniqueSelected.length === values.length
        ? []
        : uniqueSelected;
    this.onChange(this.getState());
  }

  private addYearRange(): void {
    const [minimum, maximum] = this.options.yearRange;
    const fieldset = document.createElement("fieldset");
    fieldset.className = "filter-group year-filter";
    const legend = document.createElement("legend");
    legend.textContent = "Publication years";
    fieldset.append(legend);

    const rangeLabel = document.createElement("div");
    rangeLabel.className = "year-range-label";
    const current = document.createElement("span");
    current.textContent = `${this.state.yearRange[0]} – ${this.state.yearRange[1]}`;
    rangeLabel.append(current);

    for (const [index, labelText] of ["From", "To"].entries()) {
      const label = document.createElement("label");
      label.className = "range-control";
      const accessibleLabel = document.createElement("span");
      accessibleLabel.textContent = labelText;
      const input = document.createElement("input");
      input.type = "range";
      input.min = String(minimum);
      input.max = String(maximum);
      input.step = "1";
      input.value = String(this.state.yearRange[index]);
      input.setAttribute("aria-label", `${labelText} publication year`);
      input.addEventListener("input", () => {
        const selectedYear = Number(input.value);
        const [from, to] = this.state.yearRange;
        this.state.yearRange = index === 0
          ? [Math.min(selectedYear, to), to]
          : [from, Math.max(selectedYear, from)];
        current.textContent = `${this.state.yearRange[0]} – ${this.state.yearRange[1]}`;
        this.onChange(this.getState());
      });
      label.append(accessibleLabel, input);
      fieldset.append(label);
    }
    fieldset.prepend(rangeLabel);
    this.container.append(fieldset);
  }
}
