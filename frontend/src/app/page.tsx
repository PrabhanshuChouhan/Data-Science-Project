
"use client";

import type React from "react";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  BarChart3,
  BrainCircuit,
  ChevronRight,
  CircleDot,
  Clock3,
  FileText,
  GitBranch,
  Network,
  RefreshCw,
  ShieldCheck,
  Users,
} from "lucide-react";

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

// ---------------------------------------------------------
// TYPES AND CONFIGURATION
// ---------------------------------------------------------

type Row = Record<string, unknown>;
type Dataset = Record<string, unknown>;

const DATA_FILES = [
  "overview",
  "category_shares",
  "category_drift",
  "semantic_drift",
  "topic_prevalence",
  "topic_keywords",
  "topic_examples",
  "clustering_metrics",
  "communities",
  "central_articles",
  "event_impact",
  "event_summary",
  "private_statistics",
  "privacy_metadata",
  "fairness_summary",
  "author_concentration",
  "category_by_author_cohort",
  "category_disparity",
  "cohort_distribution_divergence",
] as const;

const COLORS = [
  "#2DE2C5",
  "#8B9CFF",
  "#F6B85E",
  "#FF718B",
  "#B58CFF",
  "#55B8E8",
  "#9BD88B",
  "#E9A3D5",
];

const NAV_ITEMS = [
  { id: "overview", label: "Overview", icon: Activity },
  { id: "trends", label: "Trend Intelligence", icon: BarChart3 },
  { id: "topics", label: "Topic Discovery", icon: BrainCircuit },
  { id: "network", label: "Article Network", icon: Network },
  { id: "events", label: "Event Analysis", icon: Clock3 },
  { id: "privacy", label: "Privacy", icon: ShieldCheck },
  { id: "fairness", label: "Representation", icon: Users },
];

const MONTH_KEYS = [
  "month",
  "window",
  "window_start",
  "window_end",
  "date",
  "period",
  "timestamp",
  "time",
];

const CATEGORY_KEYS = ["category", "category_name", "label"];

const VALUE_KEYS = [
  "share",
  "proportion",
  "category_share",
  "percentage",
  "count",
  "article_count",
  "frequency",
  "value",
];

const TOPIC_KEYS = [
  "kmeans_cluster",
  "topic",
  "topic_id",
  "cluster",
  "cluster_id",
  "gmm_cluster",
];

const TOOLTIP_STYLE = {
  background: "#111B2E",
  border: "1px solid #26364E",
  borderRadius: 10,
  color: "#EAF1FF",
};

// ---------------------------------------------------------
// DATA HELPERS
// ---------------------------------------------------------

function asRows(value: unknown): Row[] {
  if (Array.isArray(value)) {
    return value.filter(
      (item): item is Row =>
        typeof item === "object" &&
        item !== null &&
        !Array.isArray(item)
    );
  }

  // Also support objects that wrap rows in a data/records/results property.
  if (typeof value === "object" && value !== null) {
    const object = value as Dataset;

    for (const key of ["data", "records", "rows", "results"]) {
      if (Array.isArray(object[key])) {
        return asRows(object[key]);
      }
    }
  }

  return [];
}

function asObject(value: unknown): Dataset {
  return typeof value === "object" &&
    value !== null &&
    !Array.isArray(value)
    ? (value as Dataset)
    : {};
}

function getValue(row: Row, keys: string[]): unknown {
  for (const key of keys) {
    if (
      row[key] !== undefined &&
      row[key] !== null &&
      row[key] !== ""
    ) {
      return row[key];
    }
  }

  return undefined;
}

function getText(value: unknown, fallback = "—"): string {
  if (value === undefined || value === null || value === "") {
    return fallback;
  }

  if (typeof value === "object") {
    return JSON.stringify(value);
  }

  return String(value);
}

function getNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }

  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);

    return Number.isFinite(parsed) ? parsed : null;
  }

  return null;
}

function formatNumber(value: unknown): string {
  const number = getNumber(value);

  return number === null
    ? "—"
    : number.toLocaleString("en-IN", {
        maximumFractionDigits: 3,
      });
}

function formatPercent(value: unknown): string {
  const number = getNumber(value);

  if (number === null) return "—";

  return `${(number <= 1 ? number * 100 : number).toFixed(1)}%`;
}

function formatDate(value: unknown): string {
  if (!value) return "—";

  const raw = String(value);
  const date = new Date(raw);

  if (Number.isNaN(date.getTime())) return raw;

  return date.toLocaleDateString("en-IN", {
    month: "short",
    year: "numeric",
  });
}

function titleCase(value: string): string {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function getMonthKey(row: Row): string {
  return getText(getValue(row, MONTH_KEYS), "");
}

function normalizeRows(
  rows: Row[],
  aliases: Record<string, string[]>
): Row[] {
  return rows.map((row) => {
    const normalized: Row = { ...row };

    for (const [target, candidates] of Object.entries(aliases)) {
      const value = getValue(row, candidates);

      if (value !== undefined) {
        normalized[target] = value;
      }
    }

    return normalized;
  });
}

function findColumn(
  rows: Row[],
  candidates: string[]
): string | undefined {
  if (!rows.length) return undefined;

  return candidates.find((key) =>
    rows.some((row) => row[key] !== undefined && row[key] !== null)
  );
}

function getDynamicColumns(
  rows: Row[],
  limit = 7
): { key: string; label: string }[] {
  if (!rows.length) return [];

  return Object.keys(rows[0])
    .slice(0, limit)
    .map((key) => ({
      key,
      label: titleCase(key),
    }));
}

// ---------------------------------------------------------
// CATEGORY DATA NORMALIZATION
// ---------------------------------------------------------

function buildCategoryChart(rows: Row[]) {
  if (!rows.length) {
    return {
      data: [] as Row[],
      categories: [] as string[],
      totals: [] as Row[],
    };
  }

  const monthKey = findColumn(rows, MONTH_KEYS);

  const categoryKey = findColumn(rows, CATEGORY_KEYS);

  const valueKey = findColumn(rows, VALUE_KEYS);

  // Long format:
  // month | category | share
  // month | category | count
  if (monthKey && categoryKey && valueKey) {
    const grouped = new Map<string, Row>();
    const totals = new Map<string, number>();
    const categories = new Set<string>();

    for (const row of rows) {
      const month = getText(row[monthKey], "");
      const category = getText(row[categoryKey], "");
      const value = getNumber(row[valueKey]);

      if (!month || !category || value === null) continue;

      categories.add(category);

      if (!grouped.has(month)) {
        grouped.set(month, { month });
      }

      grouped.get(month)![category] = value;

      totals.set(
        category,
        (totals.get(category) ?? 0) + value
      );
    }

    const categoryList = Array.from(categories).sort();

    const data = Array.from(grouped.values()).sort((a, b) =>
      getText(a.month).localeCompare(getText(b.month))
    );

    const totalRows = categoryList.map((category) => ({
      name: category,
      value: totals.get(category) ?? 0,
    }));

    return {
      data,
      categories: categoryList,
      totals: totalRows,
    };
  }

  // Wide format:
  // month | ENTERTAINMENT | POLITICS | WELLNESS ...
  if (monthKey) {
    const excluded = new Set([
      monthKey,
      "id",
      "total",
      "article_count",
      "window_count",
      "date",
    ]);

    const categories = Object.keys(rows[0]).filter(
      (key) =>
        !excluded.has(key) &&
        rows.some((row) => getNumber(row[key]) !== null)
    );

    const data = rows
      .map((row) => {
        const point: Row = {
          month: getText(row[monthKey], ""),
        };

        for (const category of categories) {
          point[category] = getNumber(row[category]);
        }

        return point;
      })
      .filter((row) => row.month)
      .sort((a, b) =>
        getText(a.month).localeCompare(getText(b.month))
      );

    const totals = categories.map((category) => ({
      name: category,
      value: rows.reduce(
        (sum, row) => sum + (getNumber(row[category]) ?? 0),
        0
      ),
    }));

    return { data, categories, totals };
  }

  // Category-only data:
  // category | count
  if (categoryKey && valueKey) {
    const totals = new Map<string, number>();

    for (const row of rows) {
      const category = getText(row[categoryKey], "");
      const value = getNumber(row[valueKey]);

      if (!category || value === null) continue;

      totals.set(
        category,
        (totals.get(category) ?? 0) + value
      );
    }

    const totalRows = Array.from(totals.entries()).map(
      ([name, value]) => ({ name, value })
    );

    return {
      data: [],
      categories: totalRows.map((row) => row.name),
      totals: totalRows,
    };
  }

  return {
    data: [],
    categories: [] as string[],
    totals: [] as Row[],
  };
}

// ---------------------------------------------------------
// GENERIC CHART HELPERS
// ---------------------------------------------------------

function getNumericChartData(
  rows: Row[],
  labelKeys: string[],
  valueKeys: string[]
) {
  return rows
    .map((row) => ({
      label: getText(getValue(row, labelKeys), ""),
      value: getNumber(getValue(row, valueKeys)),
    }))
    .filter(
      (item): item is { label: string; value: number } =>
        item.label !== "" && item.value !== null
    );
}

function buildMultiSeriesData(
  rows: Row[],
  labelKeys: string[],
  seriesKeys: string[]
) {
  const labelKey = findColumn(rows, labelKeys);

  if (!labelKey) return { data: [], series: [] as string[] };

  const series = seriesKeys.filter((key) =>
    rows.some((row) => getNumber(row[key]) !== null)
  );

  const data = rows
    .map((row) => {
      const point: Row = {
        label: getText(row[labelKey], ""),
      };

      for (const key of series) {
        point[key] = getNumber(row[key]);
      }

      return point;
    })
    .filter((row) => row.label)
    .sort((a, b) =>
      getText(a.label).localeCompare(getText(b.label))
    );

  return { data, series };
}

// ---------------------------------------------------------
// DISPLAY COMPONENTS
// ---------------------------------------------------------

function SectionHeading({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description?: string;
}) {
  return (
    <div className="section-heading">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h2>{title}</h2>
        {description && (
          <p className="section-description">{description}</p>
        )}
      </div>
    </div>
  );
}

function Panel({
  title,
  subtitle,
  children,
  className = "",
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel-heading">
        <div>
          <h3>{title}</h3>
          {subtitle && <p>{subtitle}</p>}
        </div>
      </div>

      {children}
    </section>
  );
}

function EmptyState({
  message = "No data available yet.",
}: {
  message?: string;
}) {
  return <div className="empty-state">{message}</div>;
}

function MetricCard({
  label,
  value,
  detail,
  icon: Icon,
}: {
  label: string;
  value: string;
  detail: string;
  icon: React.ElementType;
}) {
  return (
    <div className="metric-card">
      <div className="metric-top">
        <span>{label}</span>

        <div className="metric-icon">
          <Icon size={18} />
        </div>
      </div>

      <div className="metric-value">{value}</div>
      <p>{detail}</p>
    </div>
  );
}

function DataTable({
  rows,
  columns,
  emptyMessage,
}: {
  rows: Row[];
  columns: { key: string; label: string }[];
  emptyMessage?: string;
}) {
  if (!rows.length || !columns.length) {
    return <EmptyState message={emptyMessage} />;
  }

  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key}>{column.label}</th>
            ))}
          </tr>
        </thead>

        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>
              {columns.map((column) => (
                <td key={column.key}>
                  {getText(row[column.key])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------
// MAIN DASHBOARD
// ---------------------------------------------------------

export default function Home() {
  const [data, setData] = useState<Dataset>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [activeSection, setActiveSection] = useState("overview");

  // -------------------------------------------------------
  // LOAD JSON DATA
  // -------------------------------------------------------

  const loadData = useCallback(async () => {
    setLoading(true);
    setError("");

    try {
      const entries = await Promise.all(
        DATA_FILES.map(async (name) => {
          try {
            const response = await fetch(`/data/${name}.json`, {
              cache: "no-store",
            });

            if (!response.ok) {
              return [name, null] as const;
            }

            return [name, await response.json()] as const;
          } catch {
            return [name, null] as const;
          }
        })
      );

      setData(Object.fromEntries(entries));
    } catch {
      setError(
        "Could not load dashboard data. Check your exported JSON files."
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  // -------------------------------------------------------
  // EXTRACT DATASETS
  // -------------------------------------------------------

  const overview = asObject(data.overview);

  const categoryRows = asRows(data.category_shares);
  const driftRows = asRows(data.category_drift);
  const semanticRows = asRows(data.semantic_drift);
  const prevalenceRows = asRows(data.topic_prevalence);

  const keywordRows = asRows(data.topic_keywords);
  const exampleRows = asRows(data.topic_examples);
  const metricRows = asRows(data.clustering_metrics);

  const communityRows = asRows(data.communities);
  const centralRows = asRows(data.central_articles);

  const eventRows = asRows(data.event_impact);
  const eventSummaryRows = asRows(data.event_summary);

  const privateRows = asRows(data.private_statistics);
  const privacyMetadata = asObject(data.privacy_metadata);

  const fairnessSummary = asObject(data.fairness_summary);
  const authorRows = asRows(data.author_concentration);
  const cohortRows = asRows(data.category_by_author_cohort);
  const disparityRows = asRows(data.category_disparity);
  const divergenceRows = asRows(data.cohort_distribution_divergence);

  // -------------------------------------------------------
  // CATEGORY CHART
  // -------------------------------------------------------

  const categoryChart = useMemo(
    () => buildCategoryChart(categoryRows),
    [data.category_shares]
  );

  // -------------------------------------------------------
  // SEMANTIC DRIFT CHART
  // -------------------------------------------------------

  const semanticChart = useMemo(
    () =>
      getNumericChartData(
        semanticRows,
        MONTH_KEYS,
        [
          "mean_js_divergence",
          "mean_drift",
          "semantic_drift",
          "drift_score",
          "mean_js",
          "js_divergence",
          "mean_dimension_js",
          "mean_js_distance",
        ]
      ),
    [data.semantic_drift]
  );

  // -------------------------------------------------------
  // TOPIC PREVALENCE CHART
  // -------------------------------------------------------

  const prevalenceLabelKey = findColumn(prevalenceRows, MONTH_KEYS);

  const prevalenceValueKey = findColumn(prevalenceRows, [
    "share",
    "proportion",
    "prevalence",
    "count",
    "article_count",
    "topic_share",
  ]);

  const prevalenceTopicKey = findColumn(prevalenceRows, TOPIC_KEYS);

  const prevalenceChart = useMemo(() => {
    if (
      prevalenceLabelKey &&
      prevalenceValueKey &&
      prevalenceTopicKey
    ) {
      const grouped = new Map<string, Row>();
      const topics = new Set<string>();

      for (const row of prevalenceRows) {
        const month = getText(row[prevalenceLabelKey], "");
        const topic = getText(row[prevalenceTopicKey], "");
        const value = getNumber(row[prevalenceValueKey]);

        if (!month || !topic || value === null) continue;

        topics.add(topic);

        if (!grouped.has(month)) {
          grouped.set(month, { label: month });
        }

        grouped.get(month)![`Topic ${topic}`] = value;
      }

      return {
        data: Array.from(grouped.values()).sort((a, b) =>
          getText(a.label).localeCompare(getText(b.label))
        ),
        topics: Array.from(topics).map((topic) => `Topic ${topic}`),
      };
    }

    const chart = getNumericChartData(
      prevalenceRows,
      MONTH_KEYS,
      [
        "share",
        "proportion",
        "prevalence",
        "count",
        "article_count",
        "topic_share",
      ]
    );

    return {
      data: chart.map((item) => ({
        label: item.label,
        value: item.value,
      })),
      topics: [] as string[],
    };
  }, [
    data.topic_prevalence,
    prevalenceLabelKey,
    prevalenceValueKey,
    prevalenceTopicKey,
  ]);

  // -------------------------------------------------------
  // TOPIC KEYWORDS
  // -------------------------------------------------------

  const topicKey = findColumn(keywordRows, TOPIC_KEYS);

  const topicCounts = useMemo(() => {
    if (!topicKey) return [];

    return Array.from(
      new Set(
        keywordRows.map((row) => getText(row[topicKey], ""))
      )
    ).filter(Boolean);
  }, [data.topic_keywords, topicKey]);

  // -------------------------------------------------------
  // NORMALIZE TABLE DATA
  // -------------------------------------------------------

  const normalizedDriftRows = normalizeRows(driftRows, {
    month: MONTH_KEYS,
    category: CATEGORY_KEYS,
    share_change: [
      "share_change",
      "difference",
      "absolute_change",
      "change",
      "delta",
    ],
    adjusted_p_value: [
      "adjusted_p_value",
      "p_adjusted",
      "pvalue_adjusted",
      "fdr_p_value",
      "p_adj",
    ],
  });

  const normalizedPrivateRows = normalizeRows(privateRows, {
    month: MONTH_KEYS,
    category: CATEGORY_KEYS,
    noisy_count: [
      "noisy_count",
      "count_noisy",
      "private_count",
      "noisy_value",
    ],
    noisy_share: [
      "noisy_share",
      "share_noisy",
      "private_share",
      "noisy_proportion",
    ],
  });

  const normalizedExamples = normalizeRows(exampleRows, {
    kmeans_cluster: TOPIC_KEYS,
    headline: [
      "headline",
      "title",
      "article_title",
      "text",
    ],
    text: [
      "text",
      "article_text",
      "description",
      "short_description",
      "headline",
      "title",
    ],
    category: CATEGORY_KEYS,
  });

  const normalizedCentralArticles = normalizeRows(centralRows, {
    id: [
      "id",
      "article_id",
      "node_id",
    ],
    pagerank: [
      "pagerank",
      "page_rank",
      "pageRank",
    ],
    degree_centrality: [
      "degree_centrality",
      "degree",
      "centrality",
    ],
    weighted_degree: [
      "weighted_degree",
      "weighted_degree_centrality",
      "strength",
    ],
  });

  const normalizedCommunities = normalizeRows(communityRows, {
    community: [
      "community",
      "community_id",
      "partition",
      "id",
    ],
    size: [
      "size",
      "articles",
      "article_count",
      "n_articles",
      "node_count",
    ],
    top_categories: [
      "top_categories",
      "categories",
      "category_distribution",
    ],
    top_keywords: [
      "top_keywords",
      "keywords",
      "terms",
      "community_keywords",
    ],
  });

  const normalizedSemanticRows = normalizeRows(semanticRows, {
    month: MONTH_KEYS,
    drift_score: [
      "mean_js_divergence",
      "mean_drift",
      "semantic_drift",
      "drift_score",
      "mean_js",
      "js_divergence",
      "mean_dimension_js",
      "mean_js_distance",
    ],
  });

  const normalizedPrevalenceRows = normalizeRows(prevalenceRows, {
    month: MONTH_KEYS,
    topic: TOPIC_KEYS,
    value: [
      "share",
      "proportion",
      "prevalence",
      "count",
      "article_count",
      "topic_share",
    ],
  });

  // -------------------------------------------------------
  // CLUSTERING METRICS
  // -------------------------------------------------------

  const normalizedMetricRows = metricRows.map((row) => ({
    ...row,
    model: getValue(row, [
      "model",
      "algorithm",
      "method",
      "clustering_method",
      "name",
    ]),
    silhouette: getValue(row, [
      "silhouette",
      "silhouette_score",
    ]),
    davies_bouldin: getValue(row, [
      "davies_bouldin",
      "davies_bouldin_score",
      "db_score",
    ]),
    calinski_harabasz: getValue(row, [
      "calinski_harabasz",
      "calinski_harabasz_score",
      "ch_score",
    ]),
    nmi: getValue(row, [
      "nmi",
      "normalized_mutual_information",
      "normalized_mutual_info",
    ]),
  }));

  // -------------------------------------------------------
  // OVERVIEW VALUES
  // -------------------------------------------------------

  const dateRange = asObject(overview.date_range);

  const startDate = dateRange.start;
  const endDate = dateRange.end;

  // -------------------------------------------------------
  // NAVIGATION
  // -------------------------------------------------------

  const scrollToSection = (id: string) => {
    setActiveSection(id);

    document.getElementById(id)?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  };

  // -------------------------------------------------------
  // LOADING SCREEN
  // -------------------------------------------------------

  if (loading) {
    return (
      <main className="loading-screen">
        <div className="loading-mark">
          <Activity size={30} />
        </div>

        <h2>Streaming Intelligence Engine</h2>
        <p>Loading analytical results...</p>
      </main>
    );
  }

  // -------------------------------------------------------
  // DASHBOARD
  // -------------------------------------------------------

  return (
    <main className="app-shell">

      {/* SIDEBAR */}

      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Activity size={22} />
          </div>

          <div>
            <strong>STREAMLINE</strong>
            <span>DATA INTELLIGENCE</span>
          </div>
        </div>

        <div className="sidebar-label">WORKSPACE</div>

        <nav className="nav-list">
          {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => scrollToSection(id)}
              className={`nav-item ${
                activeSection === id ? "active" : ""
              }`}
            >
              <Icon size={18} />
              <span>{label}</span>

              {activeSection === id && (
                <ChevronRight size={15} />
              )}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="status-dot" />

          <div>
            <strong>Analysis workspace</strong>
            <span>Historical dataset mode</span>
          </div>
        </div>
      </aside>

      {/* MAIN CONTENT */}

      <div className="main-content">

        {/* TOPBAR */}

        <header className="topbar">
          <div>
            <span className="breadcrumb">WORKSPACE / </span>
            <strong>INTELLIGENCE OVERVIEW</strong>
          </div>

          <div className="topbar-right">
            <span className="dataset-status">
              <span className="status-dot" />
              DATA LOADED
            </span>

            <button
              className="refresh-button"
              onClick={() => void loadData()}
            >
              <RefreshCw size={15} />
              Refresh
            </button>
          </div>
        </header>

        <div className="dashboard">

          {/* =================================================
              OVERVIEW
          ================================================= */}

          <section
            id="overview"
            className="dashboard-section"
          >
            <div className="hero">
              <div>
                <p className="eyebrow">
                  NEWS STREAM ANALYTICS / HISTORICAL MODE
                </p>

                <h1>
                  Streaming Data
                  <br />
                  <span>Intelligence Engine</span>
                </h1>

                <p className="hero-description">
                  Discover emerging topics, track changing narratives,
                  identify influential articles, and investigate shifts
                  in news coverage.
                </p>

                <div className="hero-meta">
                  <span>
                    <Clock3 size={14} />
                    {formatDate(startDate)} — {formatDate(endDate)}
                  </span>

                  <span>
                    <FileText size={14} />
                    Historical news dataset
                  </span>
                </div>
              </div>

              <div className="hero-graphic">
                <div className="orbit orbit-one" />
                <div className="orbit orbit-two" />
                <div className="orbit orbit-three" />

                <div className="orbit-core">
                  <Activity size={38} />
                </div>

                <span className="orbit-point point-one" />
                <span className="orbit-point point-two" />
                <span className="orbit-point point-three" />
              </div>
            </div>

            {error && (
              <div className="error-banner">{error}</div>
            )}

            <SectionHeading
              eyebrow="SYSTEM SNAPSHOT"
              title="Dataset overview"
              description="A summary of the data and analysis currently available."
            />

            <div className="metrics-grid">
              <MetricCard
                label="Total articles"
                value={formatNumber(overview.total_articles)}
                detail="Articles in the processed dataset"
                icon={FileText}
              />

              <MetricCard
                label="Categories"
                value={formatNumber(overview.total_categories)}
                detail="News categories retained"
                icon={CircleDot}
              />

              <MetricCard
                label="Time windows"
                value={formatNumber(overview.total_windows)}
                detail="Distinct chronological windows"
                icon={Clock3}
              />

              <MetricCard
                label="Discovered topics"
                value={formatNumber(overview.total_topics)}
                detail="Clustering groups identified"
                icon={BrainCircuit}
              />

              <MetricCard
                label="Graph nodes"
                value={formatNumber(overview.graph_nodes)}
                detail="Articles represented in the graph"
                icon={Network}
              />
            </div>

            <div className="two-column">

              {/* CATEGORY DISTRIBUTION */}

              <Panel
                title="Category distribution"
                subtitle="Number of articles in each news category"
              >
                {categoryChart.totals.length > 0 ? (
                  <div className="chart-wrap">
                    <ResponsiveContainer
                      width="100%"
                      height={280}
                    >
                      <PieChart>
                        <Pie
                          data={categoryChart.totals}
                          dataKey="value"
                          nameKey="name"
                          innerRadius={65}
                          outerRadius={100}
                          paddingAngle={3}
                        >
                          {categoryChart.totals.map(
                            (item, index) => (
                              <Cell key={String(item.name)} fill={COLORS[index % COLORS.length]} />
                            )
                          )}
                        </Pie>

                        <Tooltip
                          contentStyle={TOOLTIP_STYLE}
                          formatter={(value) => [
                            formatNumber(value),
                            "Articles",
                          ]}
                        />

                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <EmptyState message="Category distribution data is unavailable." />
                )}
              </Panel>

              {/* CLUSTERING EVALUATION */}

              <Panel
                title="Clustering evaluation"
                subtitle="Internal and external clustering metrics"
              >
                <DataTable
                  rows={normalizedMetricRows}
                  columns={[
                    { key: "model", label: "Model" },
                    { key: "silhouette", label: "Silhouette" },
                    {
                      key: "davies_bouldin",
                      label: "Davies-Bouldin",
                    },
                    {
                      key: "calinski_harabasz",
                      label: "Calinski-Harabasz",
                    },
                    { key: "nmi", label: "NMI" },
                  ]}
                />
              </Panel>
            </div>
          </section>

          {/* =================================================
              TREND INTELLIGENCE
          ================================================= */}

          <section
            id="trends"
            className="dashboard-section"
          >
            <SectionHeading
              eyebrow="TEMPORAL INTELLIGENCE"
              title="Trend detection"
              description="Track category popularity and identify statistically significant changes over time."
            />

            {/* CATEGORY SHARE */}

            <Panel
              title="Category share over time"
              subtitle="Monthly category proportions across the observation period"
            >
              {categoryChart.data.length > 0 &&
              categoryChart.categories.length > 0 ? (
                <div className="chart-wrap">
                  <ResponsiveContainer
                    width="100%"
                    height={350}
                  >
                    <AreaChart data={categoryChart.data}>
                      <CartesianGrid
                        stroke="#26364E"
                        strokeDasharray="3 3"
                      />

                      <XAxis
                        dataKey="month"
                        stroke="#8493A9"
                        tick={{ fontSize: 11 }}
                      />

                      <YAxis
                        stroke="#8493A9"
                        tick={{ fontSize: 11 }}
                        tickFormatter={(value) =>
                          value <= 1
                            ? `${(value * 100).toFixed(0)}%`
                            : value
                        }
                      />

                      <Tooltip
                        contentStyle={TOOLTIP_STYLE}
                      />

                      <Legend />

                      {categoryChart.categories.map(
                        (category, index) => (
                          <Area
                            key={category}
                            type="monotone"
                            dataKey={category}
                            stackId="1"
                            stroke={COLORS[index % COLORS.length]}
                            fill={COLORS[index % COLORS.length]}
                            fillOpacity={0.12}
                            strokeWidth={2}
                          />
                        )
                      )}
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <EmptyState message="Category time-series data is unavailable." />
              )}
            </Panel>

            <div className="two-column">

              {/* CATEGORY DRIFT */}

              <Panel
                title="Category drift detections"
                subtitle="Statistical changes between adjacent time windows"
              >
                <DataTable
                  rows={normalizedDriftRows.slice(0, 30)}
                  columns={[
                    { key: "month", label: "Window" },
                    { key: "category", label: "Category" },
                    {
                      key: "share_change",
                      label: "Share change",
                    },
                    {
                      key: "adjusted_p_value",
                      label: "Adjusted p-value",
                    },
                  ]}
                />
              </Panel>

              {/* SEMANTIC DRIFT */}

              <Panel
                title="Semantic drift"
                subtitle="Changes in embedding distributions over time"
              >
                {semanticChart.length > 0 ? (
                  <div className="chart-wrap">
                    <ResponsiveContainer
                      width="100%"
                      height={270}
                    >
                      <LineChart data={semanticChart}>
                        <CartesianGrid
                          stroke="#26364E"
                          strokeDasharray="3 3"
                        />

                        <XAxis
                          dataKey="label"
                          stroke="#8493A9"
                          tick={{ fontSize: 10 }}
                        />

                        <YAxis
                          stroke="#8493A9"
                          tick={{ fontSize: 11 }}
                        />

                        <Tooltip
                          contentStyle={TOOLTIP_STYLE}
                        />

                        <Line
                          type="monotone"
                          dataKey="value"
                          name="Semantic drift"
                          stroke="#2DE2C5"
                          strokeWidth={2}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <EmptyState message="No compatible semantic drift score column was found." />
                )}
              </Panel>
            </div>

            {/* TOPIC PREVALENCE */}

            <Panel
              title="Topic prevalence over time"
              subtitle="How discovered clusters change in representation"
            >
              {prevalenceChart.data.length > 0 ? (
                <div className="chart-wrap">
                  <ResponsiveContainer
                    width="100%"
                    height={300}
                  >
                    <BarChart data={prevalenceChart.data}>
                      <CartesianGrid
                        stroke="#26364E"
                        strokeDasharray="3 3"
                      />

                      <XAxis
                        dataKey="label"
                        stroke="#8493A9"
                        tick={{ fontSize: 11 }}
                      />

                      <YAxis stroke="#8493A9" />

                      <Tooltip
                        contentStyle={TOOLTIP_STYLE}
                      />

                      <Legend />

                      {prevalenceChart.topics.length > 0 ? (
                        prevalenceChart.topics.map(
                          (topic, index) => (
                            <Bar
                              key={topic}
                              dataKey={topic}
                              fill={COLORS[index % COLORS.length]}
                              radius={[4, 4, 0, 0]}
                            />
                          )
                        )
                      ) : (
                        <Bar
                          dataKey="value"
                          name="Prevalence"
                          fill="#8B9CFF"
                          radius={[5, 5, 0, 0]}
                        />
                      )}
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <EmptyState message="Topic prevalence data is unavailable." />
              )}
            </Panel>
          </section>

          {/* =================================================
              TOPIC DISCOVERY
          ================================================= */}

          <section
            id="topics"
            className="dashboard-section"
          >
            <SectionHeading
              eyebrow="UNSUPERVISED LEARNING"
              title="Topic discovery"
              description="Explore the clusters discovered from article text embeddings and their representative keywords."
            />

            <div className="topic-grid">
              {topicCounts.length > 0 ? (
                topicCounts.map((topic, index) => {
                  const matching = keywordRows.filter(
                    (row) =>
                      getText(row[topicKey!]) === topic
                  );

                  return (
                    <div
                      className="topic-card"
                      key={topic}
                    >
                      <div className="topic-card-top">
                        <span className="topic-number">
                          TOPIC {topic}
                        </span>

                        <span
                          className="topic-dot"
                          style={{
                            background:
                              COLORS[index % COLORS.length],
                          }}
                        />
                      </div>

                      <div className="keyword-list">
                        {matching.slice(0, 10).map(
                          (row, wordIndex) => {
                            const word = getText(
                              getValue(row, [
                                "keyword",
                                "term",
                                "word",
                                "top_keyword",
                              ]),
                              ""
                            );

                            return word ? (
                              <span key={wordIndex}>
                                {word}
                              </span>
                            ) : null;
                          }
                        )}
                      </div>
                    </div>
                  );
                })
              ) : (
                <EmptyState message="Topic keyword data is not available." />
              )}
            </div>

            {/* REPRESENTATIVE ARTICLES */}

            <Panel
              title="Representative articles"
              subtitle="Examples associated with discovered topics"
            >
              <DataTable
                rows={normalizedExamples.slice(0, 30)}
                columns={[
                  { key: "kmeans_cluster", label: "Topic" },
                  { key: "headline", label: "Headline" },
                  { key: "text", label: "Article text" },
                  { key: "category", label: "Category" },
                ]}
              />
            </Panel>
          </section>

          {/* =================================================
              ARTICLE NETWORK
          ================================================= */}

          <section
            id="network"
            className="dashboard-section"
          >
            <SectionHeading
              eyebrow="RELATIONSHIP ANALYSIS"
              title="Article network"
              description="Explore influential articles and communities formed by content similarity."
            />

            <div className="two-column">

              {/* INFLUENTIAL ARTICLES */}

              <Panel
                title="Influential articles"
                subtitle="Ranked by graph centrality or PageRank"
              >
                <DataTable
                  rows={normalizedCentralArticles.slice(0, 25)}
                  columns={[
                    { key: "id", label: "Article ID" },
                    { key: "pagerank", label: "PageRank" },
                    {
                      key: "degree_centrality",
                      label: "Degree",
                    },
                    {
                      key: "weighted_degree",
                      label: "Weighted degree",
                    },
                  ]}
                />
              </Panel>

              {/* COMMUNITIES */}

              <Panel
                title="Detected communities"
                subtitle="Groups of closely related articles"
              >
                <DataTable
                  rows={normalizedCommunities.slice(0, 25)}
                  columns={[
                    { key: "community", label: "Community" },
                    { key: "size", label: "Articles" },
                    {
                      key: "top_categories",
                      label: "Top categories",
                    },
                    {
                      key: "top_keywords",
                      label: "Keywords",
                    },
                  ]}
                />
              </Panel>
            </div>
          </section>

          {/* =================================================
              EVENT ANALYSIS
          ================================================= */}

          <section
            id="events"
            className="dashboard-section"
          >
            <SectionHeading
              eyebrow="EVENT-BASED ANALYSIS"
              title="Event impact"
              description="Examine changes in category trends around documented events using interrupted time-series analysis."
            />

            <Panel
              title="Event impact estimates"
              subtitle="Estimated level and trend changes around each event"
            >
              <DataTable
                rows={eventRows}
                columns={getDynamicColumns(eventRows, 10)}
              />
            </Panel>

            <Panel
              title="Event summary"
              subtitle="Aggregated results from the event analysis"
            >
              <DataTable
                rows={eventSummaryRows}
                columns={getDynamicColumns(eventSummaryRows, 8)}
              />
            </Panel>

            <p className="method-note">
              Interpretation note: these are observational time-series
              associations. They do not establish that an event caused
              a change in news coverage.
            </p>
          </section>

          {/* =================================================
              PRIVACY
          ================================================= */}

          <section
            id="privacy"
            className="dashboard-section"
          >
            <SectionHeading
              eyebrow="PRIVACY ENGINEERING"
              title="Differential privacy"
              description="Inspect privacy-protected category statistics and the release parameters."
            />

            <div className="metrics-grid metrics-grid-small">
              <MetricCard
                label="Privacy budget"
                value={getText(
                  getValue(privacyMetadata, [
                    "epsilon_per_month",
                    "epsilon",
                  ])
                )}
                detail="Configured epsilon value"
                icon={ShieldCheck}
              />

              <MetricCard
                label="Composition"
                value={getText(
                  getValue(privacyMetadata, [
                    "composition",
                    "composition_method",
                    "accounting_method",
                  ]),
                  "See metadata"
                )}
                detail="Privacy accounting method"
                icon={GitBranch}
              />

              <MetricCard
                label="Adjacency"
                value={getText(
                  getValue(privacyMetadata, [
                    "adjacency",
                    "adjacency_definition",
                  ]),
                  "See metadata"
                )}
                detail="Record-level privacy definition"
                icon={Users}
              />
            </div>

            <Panel
              title="Private category statistics"
              subtitle="Noisy category counts and shares released by the privacy mechanism"
            >
              <DataTable
                rows={normalizedPrivateRows.slice(0, 40)}
                columns={[
                  { key: "month", label: "Month" },
                  { key: "category", label: "Category" },
                  {
                    key: "noisy_count",
                    label: "Noisy count",
                  },
                  {
                    key: "noisy_share",
                    label: "Noisy share",
                  },
                ]}
              />
            </Panel>

            <Panel
              title="Privacy metadata"
              subtitle="Configuration and accounting details"
            >
              <pre className="metadata-block">
                {JSON.stringify(
                  privacyMetadata,
                  null,
                  2
                )}
              </pre>
            </Panel>

            <p className="method-note">
              Privacy protection applies only to the statistics
              released by this mechanism. It does not automatically
              privatize article text, embeddings, graph outputs,
              or other dashboard analyses.
            </p>
          </section>

          {/* =================================================
              REPRESENTATION / FAIRNESS
          ================================================= */}

          <section
            id="fairness"
            className="dashboard-section"
          >
            <SectionHeading
              eyebrow="REPRESENTATION AUDIT"
              title="Author and category representation"
              description="Analyze byline concentration and differences in category coverage across author activity cohorts."
            />

            <Panel
              title="Representation summary"
              subtitle="Summary generated by the representation audit"
            >
              <pre className="metadata-block">
                {JSON.stringify(
                  fairnessSummary,
                  null,
                  2
                )}
              </pre>
            </Panel>

            <Panel
              title="Author concentration"
              subtitle="Distribution of published articles across authors"
            >
              <DataTable
                rows={authorRows}
                columns={getDynamicColumns(authorRows, 8)}
              />
            </Panel>

            <div className="two-column">
              <Panel
                title="Category disparity"
                subtitle="Differences in category representation across cohorts"
              >
                <DataTable
                  rows={disparityRows.slice(0, 30)}
                  columns={getDynamicColumns(disparityRows, 7)}
                />
              </Panel>

              <Panel
                title="Cohort distribution divergence"
                subtitle="Distribution distance from the overall author population"
              >
                <DataTable
                  rows={divergenceRows}
                  columns={getDynamicColumns(divergenceRows, 7)}
                />
              </Panel>
            </div>

            <Panel
              title="Category by author cohort"
              subtitle="Category composition across low, medium, and high activity groups"
            >
              <DataTable
                rows={cohortRows.slice(0, 40)}
                columns={getDynamicColumns(cohortRows, 8)}
              />
            </Panel>

            <p className="method-note">
              This is a descriptive representation audit based on
              byline activity. It does not measure protected demographic
              groups or prove discrimination.
            </p>
          </section>

          {/* FOOTER */}

          <footer className="footer">
            <div className="footer-brand">
              <Activity size={16} />
              STREAMLINE DATA INTELLIGENCE
            </div>

            <span>
              Historical analysis · Statistical and ML-based insights
            </span>
          </footer>
        </div>
      </div>
    </main>
  );
}
