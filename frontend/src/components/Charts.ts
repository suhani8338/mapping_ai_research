import * as echarts from "echarts";
import type { ECharts, EChartsOption } from "echarts";
import type {
  CountryMetricRow,
  MetricRow,
  PipelineDropoffRow,
} from "../db/queries";

const COLORS = {
  applied: "#e3815d",
  core: "#4b82a5",
  bridge: "#7d70b2",
  female: "#c75d81",
  male: "#4b82a5",
  nonBinary: "#aa85c7",
  unknown: "#e6e9ed",
  text: "#34424f",
  muted: "#77838e",
  grid: "#e9edf0",
};

const CLUSTER_COLORS: Record<string, string> = {
  applied_socio_technical: COLORS.applied,
  core_infrastructure: COLORS.core,
  methodological_bridge: COLORS.bridge,
};

function label(value: string): string {
  const acronyms: Record<string, string> = { ai: "AI", nlp: "NLP", edtech: "EdTech" };
  return value.split("_").map((word) =>
    acronyms[word] ?? `${word.charAt(0).toUpperCase()}${word.slice(1)}`,
  ).join(" ");
}

function formatPercent(value: number | null): string {
  return value === null ? "N/A" : `${(value * 100).toFixed(1)}%`;
}

function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[character]!);
}

export class Charts {
  private readonly instances = new Map<string, ECharts>();
  private readonly resizeObserver: ResizeObserver;

  constructor() {
    this.resizeObserver = new ResizeObserver(() => this.resize());
    for (const id of ["subfield-chart", "dropoff-chart", "coverage-chart", "country-chart"]) {
      const element = document.getElementById(id);
      if (element) this.resizeObserver.observe(element);
    }
  }

  render(
    metrics: MetricRow[],
    dropoff: PipelineDropoffRow[],
    countries: CountryMetricRow[],
  ): void {
    this.renderSubfieldComparison("subfield-chart", metrics);
    this.renderDumbbellDropoff("dropoff-chart", dropoff);
    this.renderCoverageChart("coverage-chart", metrics);
    this.renderCountryChart("country-chart", countries);
  }

  renderSubfieldComparison(containerId: string, data: MetricRow[]): void {
    const rows = [...data].sort((a, b) =>
      a.macro_cluster.localeCompare(b.macro_cluster) || a.subfield.localeCompare(b.subfield),
    );
    if (rows.length === 0) {
      this.setOption(containerId, {}, "No subfield records match these filters.");
      return;
    }
    const subfields = rows.map((row) => row.subfield);
    const clusters = [...new Set(rows.map((row) => row.macro_cluster))];
    const options: EChartsOption = {
      color: clusters.map((cluster) => CLUSTER_COLORS[cluster] ?? COLORS.muted),
      grid: { left: 190, right: 24, top: 24, bottom: 42, containLabel: false },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        formatter: (items) => {
          const itemList = Array.isArray(items) ? items : [items];
          const index = Number(itemList[0]?.dataIndex ?? 0);
          const row = rows[index];
          if (!row) return "";
          return `<strong>${escapeHtml(label(row.subfield))}</strong><br/>` +
            `Macro-cluster: ${escapeHtml(label(row.macro_cluster))}<br/>` +
            `Female: ${row.female_pct.toFixed(1)}% of classified authors<br/>` +
            `Classified author count: ${row.known_authors.toLocaleString()}<br/>` +
            `Total authors: ${row.total_authors.toLocaleString()}`;
        },
      },
      legend: { top: 0, textStyle: { color: COLORS.text } },
      xAxis: {
        type: "value",
        min: 0,
        max: 100,
        name: "Female authors among classified (%)",
        nameLocation: "middle",
        nameGap: 30,
        axisLabel: { formatter: "{value}%" },
        splitLine: { lineStyle: { color: COLORS.grid } },
      },
      yAxis: {
        type: "category",
        data: subfields.map(label),
        inverse: true,
        axisLabel: { color: COLORS.text, width: 170, overflow: "truncate" },
        axisTick: { show: false },
        axisLine: { show: false },
      },
      series: clusters.map((cluster) => ({
        name: label(cluster),
        type: "bar",
        barMaxWidth: 18,
        data: rows.map((row) => row.macro_cluster === cluster ? row.female_pct : null),
        itemStyle: { borderRadius: [0, 5, 5, 0] },
      })),
    };
    this.setOption(containerId, options, "No subfield records match these filters.");
  }

  renderDumbbellDropoff(containerId: string, data: PipelineDropoffRow[]): void {
    const rows = data.filter((row) => row.p_first !== null && row.p_last !== null);
    if (rows.length === 0) {
      this.setOption(containerId, {}, "No first/last-author records match these filters.");
      return;
    }
    const subfields = rows.map((row) => row.subfield);
    const connectors = rows.map((row, index) => ({
      coords: [
        [row.p_first! * 100, index],
        [row.p_last! * 100, index],
      ],
    }));
    const options: EChartsOption = {
      grid: { left: 190, right: 24, top: 28, bottom: 42 },
      tooltip: {
        trigger: "item",
        formatter: (item) => {
          const itemParams = Array.isArray(item) ? item[0] : item;
          const row = rows[Number(itemParams?.dataIndex ?? 0)];
          if (!row) return "";
          return `<strong>${escapeHtml(label(row.subfield))}</strong><br/>` +
            `First author: ${formatPercent(row.p_first)} (n=${row.first_known_authors})<br/>` +
            `Last author: ${formatPercent(row.p_last)} (n=${row.last_known_authors})<br/>` +
            `First − last: ${row.delta === null ? "N/A" : `${(row.delta * 100).toFixed(1)} pp`}`;
        },
      },
      legend: { top: 0, data: ["First author", "Last author"] },
      xAxis: {
        type: "value",
        min: 0,
        max: 100,
        name: "Female authors among classified (%)",
        nameLocation: "middle",
        nameGap: 30,
        axisLabel: { formatter: "{value}%" },
        splitLine: { lineStyle: { color: COLORS.grid } },
      },
      yAxis: {
        type: "category",
        data: subfields.map(label),
        inverse: true,
        axisLabel: { color: COLORS.text, width: 170, overflow: "truncate" },
        axisTick: { show: false },
        axisLine: { show: false },
      },
      series: [
        {
          type: "lines",
          coordinateSystem: "cartesian2d",
          data: connectors,
          polyline: false,
          symbol: ["none", "arrow"],
          symbolSize: 8,
          lineStyle: { color: "#acb6bd", width: 2, opacity: 0.8 },
          silent: true,
          tooltip: { show: false },
        },
        {
          name: "First author",
          type: "scatter",
          data: rows.map((row, index) => [row.p_first! * 100, index]),
          symbolSize: 10,
          itemStyle: { color: COLORS.female },
        },
        {
          name: "Last author",
          type: "scatter",
          data: rows.map((row, index) => [row.p_last! * 100, index]),
          symbolSize: 10,
          itemStyle: { color: COLORS.core },
        },
      ],
    };
    this.setOption(containerId, options, "No first/last-author records match these filters.");
  }

  renderCoverageChart(containerId: string, data: MetricRow[]): void {
    const rows = [...data].sort((a, b) => a.subfield.localeCompare(b.subfield));
    if (rows.length === 0) {
      this.setOption(containerId, {}, "No coverage records match these filters.");
      return;
    }
    const options: EChartsOption = {
      grid: { left: 190, right: 24, top: 18, bottom: 42 },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        formatter: (items) => {
          const itemList = Array.isArray(items) ? items : [items];
          const index = Number(itemList[0]?.dataIndex ?? 0);
          const row = rows[index];
          if (!row) return "";
          return `<strong>${escapeHtml(label(row.subfield))}</strong><br/>` +
            itemList.map((item) => `${escapeHtml(String(item.seriesName))}: ` +
              `${Number(item.value).toFixed(1)}%`).join("<br/>") +
            `<br/>Total authors: ${row.total_authors.toLocaleString()}`;
        },
      },
      legend: { top: 0 },
      xAxis: {
        type: "value",
        min: 0,
        max: 100,
        axisLabel: { formatter: "{value}%" },
        splitLine: { lineStyle: { color: COLORS.grid } },
      },
      yAxis: {
        type: "category",
        data: rows.map((row) => label(row.subfield)),
        inverse: true,
        axisLabel: { color: COLORS.text, width: 170, overflow: "truncate" },
        axisTick: { show: false },
        axisLine: { show: false },
      },
      series: [
        {
          name: "Female",
          type: "bar",
          stack: "coverage",
          data: rows.map((row) => row.female_total_pct),
          itemStyle: { color: COLORS.female },
        },
        {
          name: "Male",
          type: "bar",
          stack: "coverage",
          data: rows.map((row) => row.male_pct),
          itemStyle: { color: COLORS.male },
        },
        {
          name: "Non-binary",
          type: "bar",
          stack: "coverage",
          data: rows.map((row) => row.non_binary_pct),
          itemStyle: { color: COLORS.nonBinary },
        },
        {
          name: "Unknown",
          type: "bar",
          stack: "coverage",
          data: rows.map((row) => row.unknown_pct),
          itemStyle: { color: COLORS.unknown, borderColor: "#cbd2d8", borderWidth: 1 },
        },
      ],
    };
    this.setOption(containerId, options, "No coverage records match these filters.");
  }

  private renderCountryChart(containerId: string, data: CountryMetricRow[]): void {
    const rows = [...data].sort((a, b) => a.total_authors - b.total_authors);
    if (rows.length === 0) {
      this.setOption(containerId, {
        title: {
          text: "No verified country codes in this selection",
          left: "center",
          top: "middle",
          textStyle: { color: COLORS.muted, fontSize: 14, fontWeight: "normal" },
        },
      });
      return;
    }
    const options: EChartsOption = {
      grid: { left: 56, right: 56, top: 24, bottom: 44 },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        formatter: (items) => {
          const itemList = Array.isArray(items) ? items : [items];
          const index = Number(itemList[0]?.dataIndex ?? 0);
          const row = rows[index];
          return row
            ? `<strong>${escapeHtml(row.country_code)}</strong><br/>` +
              `Authors: ${row.total_authors.toLocaleString()}<br/>` +
              `Female among classified: ${row.female_pct.toFixed(1)}%<br/>` +
              `Classified: ${row.known_authors.toLocaleString()}`
            : "";
        },
      },
      xAxis: {
        type: "category",
        data: rows.map((row) => row.country_code),
        axisTick: { show: false },
        axisLine: { lineStyle: { color: COLORS.grid } },
      },
      yAxis: {
        type: "value",
        name: "Authors",
        splitLine: { lineStyle: { color: COLORS.grid } },
      },
      series: [{
        type: "bar",
        data: rows.map((row) => ({
          value: row.total_authors,
          itemStyle: { color: COLORS.core },
        })),
        barMaxWidth: 42,
        itemStyle: { borderRadius: [5, 5, 0, 0] },
      }],
    };
    this.setOption(containerId, options);
  }

  private setOption(containerId: string, option: EChartsOption, emptyText?: string): void {
    const element = document.getElementById(containerId);
    if (!element) throw new Error(`Missing chart container "${containerId}"`);
    const chart = this.instances.get(containerId) ?? echarts.init(element);
    this.instances.set(containerId, chart);
    chart.clear();
    if (emptyText && !option.series) {
      chart.setOption({
        title: {
          text: emptyText,
          left: "center",
          top: "middle",
          textStyle: { color: COLORS.muted, fontSize: 14, fontWeight: "normal" },
        },
      });
    } else {
      chart.setOption(option);
    }
  }

  resize(): void {
    for (const chart of this.instances.values()) chart.resize();
  }
}
