
"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  BarChart3,
  BrainCircuit,
  ChevronRight,
  Clock3,
  FileText,
  GitBranch,
  Network,
  RefreshCw,
  ShieldCheck,
  TrendingUp,
  Layers3,
  Database,
} from "lucide-react";

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type Row = Record<string, any>;
type Overview = Record<string, number | string | null>;

type DashboardData = {
  overview: Overview | null;
  activity: Row[];
  growth: Row[];
  stability: Row[];
  emerging: Row[];
  categories: Row[];
  drift: Row[];
  network: Row[];
  networkEdges: number | null;
  influential: Row[];
  performance: Row[];
};

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

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
  { id: "timeline", label: "Streaming Timeline", icon: Clock3 },
  { id: "topics", label: "Topic Intelligence", icon: BrainCircuit },
  { id: "trends", label: "Trend Analysis", icon: TrendingUp },
  { id: "drift", label: "Semantic Drift", icon: GitBranch },
  { id: "network", label: "Article Network", icon: Network },
  { id: "evaluation", label: "Evaluation", icon: ShieldCheck },
];

const EMPTY_DATA: DashboardData = {
  overview: null,
  activity: [],
  growth: [],
  stability: [],
  emerging: [],
  categories: [],
  drift: [],
  network: [],
  networkEdges: null,
  influential: [],
  performance: [],
};

const TOOLTIP_STYLE = {
  backgroundColor: "#111a2c",
  border: "1px solid #26364e",
  borderRadius: "8px",
  color: "#edf3ff",
  fontSize: "11px",
};

async function fetchEndpoint(path: string) {
  const response = await fetch(`${API_BASE}${path}`, {
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(`${path}: HTTP ${response.status}`);
  }

  return response.json();
}

function asArray(value: any): Row[] {
  return Array.isArray(value) ? value : [];
}

function getNumber(value: any): number | null {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function formatNumber(value: any) {
  const number = getNumber(value);
  return number === null ? "—" : number.toLocaleString("en-IN", {
    maximumFractionDigits: 2,
  });
}

function formatPercent(value: any) {
  const number = getNumber(value);
  return number === null ? "—" : `${number.toFixed(2)}%`;
}

function getTopicLabel(
  topicId: string | number | null | undefined,
  topicName?: unknown
) {
  const name = typeof topicName === "string" ? topicName.trim() : "";
  const id = String(topicId ?? "").trim();
  return name || `Topic ${id || "Unknown"}`;
}

function findKey(row: Row | undefined, candidates: string[]) {
  if (!row) return undefined;
  return candidates.find((key) => key in row);
}

function displayKey(row: Row | undefined, candidates: string[], fallback = "—") {
  const key = findKey(row, candidates);
  return key ? String(row?.[key] ?? fallback) : fallback;
}

function SectionHeading({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description: string;
}) {
  return (
    <div className="section-heading">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h2>{title}</h2>
        <p className="section-description">{description}</p>
      </div>
    </div>
  );
}

function Panel({
  title,
  description,
  children,
}: {
  title: string;
  description?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="panel">
      <div className="panel-heading">
        <div>
          <h3>{title}</h3>
          {description && <p>{description}</p>}
        </div>
      </div>
      {children}
    </div>
  );
}

function MetricCard({
  title,
  value,
  description,
  icon: Icon,
  color,
}: {
  title: string;
  value: string;
  description: string;
  icon: any;
  color?: string;
}) {
  return (
    <div className="metric-card">
      <div className="metric-top">
        <span>{title}</span>
        <div className="metric-icon" style={{ color: color || "#2DE2C5" }}>
          <Icon size={17} />
        </div>
      </div>
      <div className="metric-value">{value}</div>
      <p>{description}</p>
    </div>
  );
}

function EmptyState({ message = "No data available for this visualization." }) {
  return <div className="empty-state">{message}</div>;
}

function DataTable({
  rows,
  columns,
  limit = 12,
}: {
  rows: Row[];
  columns?: string[];
  limit?: number;
}) {
  if (!rows.length) return <EmptyState />;

  const keys = columns || Object.keys(rows[0]).slice(0, 6);

  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            {keys.map((key) => (
              <th key={key}>{key.replaceAll("_", " ")}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, limit).map((row, index) => (
            <tr key={index}>
              {keys.map((key) => (
                <td key={key}>
                  {row[key] === null || row[key] === undefined
                    ? "—"
                    : typeof row[key] === "number"
                      ? key === "pagerank"
                        ? row[key].toFixed(6)
                        : key === "growth_percentage"
                          ? `${formatNumber(row[key])}%`
                          : formatNumber(row[key])
                      : String(row[key])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Home() {
  const [data, setData] = useState<DashboardData>(EMPTY_DATA);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState("");
  const [activeSection, setActiveSection] = useState("overview");
  const [growthWindow, setGrowthWindow] = useState("24");
  const [growthChartExpanded, setGrowthChartExpanded] = useState(false);

  const loadData = useCallback(async () => {
    setLoading(true);
    setError("");

    const endpoints = [
      fetchEndpoint("/api/overview"),
      fetchEndpoint("/api/topics/activity"),
      fetchEndpoint("/api/topics/growth"),
      fetchEndpoint("/api/topics/stability"),
      fetchEndpoint("/api/topics/emerging"),
      fetchEndpoint("/api/trends/categories"),
      fetchEndpoint("/api/analysis/semantic-drift"),
      fetchEndpoint("/api/analysis/network"),
      fetchEndpoint("/api/analysis/influential-articles"),
      fetchEndpoint("/api/evaluation/performance"),
    ];

    const results = await Promise.allSettled(endpoints);

    const value = (index: number) =>
      results[index].status === "fulfilled"
        ? results[index].value
        : null;

    const networkPayload = value(7);

    const failures = results
      .map((result, index) =>
        result.status === "rejected" ? index : -1
      )
      .filter((index) => index !== -1);

    setData({
      overview: value(0),
      activity: asArray(value(1)),
      growth: asArray(value(2)),
      stability: asArray(value(3)),
      emerging: asArray(value(4)),
      categories: asArray(value(5)),
      drift: asArray(value(6)),
      network: asArray(networkPayload?.nodes),
      networkEdges: getNumber(networkPayload?.metadata?.total_edges),
      influential: asArray(value(8)),
      performance: asArray(value(9)),
    });

    if (failures.length) {
      setError(
        `${failures.length} endpoint(s) could not be loaded. Check that the backend is running and the corresponding data files exist.`
      );
    }

    setLastUpdated(new Date().toLocaleTimeString());
    setLoading(false);
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const topicNames = useMemo(() => {
    const names: Record<string, string> = {};

    data.activity.forEach((row) => {
      const id = String(row.topic_id ?? "");
      if (id && row.topic_name) {
        names[id] = getTopicLabel(id, row.topic_name);
      }
    });

    data.growth.forEach((row) => {
      const id = String(row.topic_id ?? "");
      if (id && row.topic_name && !names[id]) {
        names[id] = getTopicLabel(id, row.topic_name);
      }
    });

    return names;
  }, [data.activity, data.growth]);

  const topicActivityChart = useMemo(() => {
    const grouped: Record<string, Row> = {};

    data.activity.forEach((row) => {
      const month = String(row.month ?? "");
      const topic = `topic_${row.topic_id}`;
      if (!month) return;

      if (!grouped[month]) grouped[month] = { month };
      grouped[month][topic] = getNumber(row.article_count) ?? 0;
    });

    return Object.values(grouped).sort((a, b) =>
      String(a.month).localeCompare(String(b.month))
    );
  }, [data.activity]);

  const topTopics = useMemo(() => {
    const totals: Record<string, number> = {};

    data.activity.forEach((row) => {
      const id = String(row.topic_id);
      totals[id] = (totals[id] || 0) + (getNumber(row.article_count) || 0);
    });

    return Object.entries(totals)
      .map(([id, count]) => ({
        id,
        count,
        name: topicNames[id] || getTopicLabel(id),
      }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 8);
  }, [data.activity, topicNames]);

  const latestActivity = useMemo(() => {
    const months = [...new Set(data.activity.map((row) => row.month))]
      .filter(Boolean)
      .sort();

    const latest = months[months.length - 1];
    return data.activity
      .filter((row) => row.month === latest)
      .map((row): Row => ({
        ...row,
        topic_name: getTopicLabel(
          row.topic_id,
          row.topic_name || topicNames[String(row.topic_id)]
        ),
      }))
      .sort(
        (a, b) =>
          (getNumber(b.article_count) || 0) -
          (getNumber(a.article_count) || 0)
      );
  }, [data.activity, topicNames]);

  const topicGrowthChart = useMemo(() => {
    const topicIds = new Set(topTopics.slice(0, 5).map((topic) => topic.id));
    const grouped: Record<string, Row> = {};

    data.growth.forEach((row) => {
      const month = String(row.month ?? "");
      const topicId = String(row.topic_id ?? "");
      if (!month || !topicIds.has(topicId)) return;

      if (!grouped[month]) grouped[month] = { month };
      grouped[month][`topic_${topicId}`] = getNumber(row.growth_percentage);
    });

    return Object.values(grouped).sort((a, b) =>
      String(a.month).localeCompare(String(b.month))
    );
  }, [data.growth, topTopics]);

  const visibleTopicGrowth = useMemo(
    () =>
      growthWindow === "all"
        ? topicGrowthChart
        : topicGrowthChart.slice(-Number(growthWindow)),
    [growthWindow, topicGrowthChart]
  );

  const latestTopicGrowth = useMemo(() => {
    const latestMonth = data.growth.reduce((latest, row) => {
      const month = String(row.month ?? "");
      return month > latest ? month : latest;
    }, "");

    return data.growth
      .filter((row) => String(row.month ?? "") === latestMonth)
      .map((row): Row => {
        const previous = getNumber(row.previous_articles) ?? 0;
        const current = getNumber(row.current_articles) ?? 0;

        return {
          ...row,
          topic_name: getTopicLabel(
            row.topic_id,
            row.topic_name || topicNames[String(row.topic_id)]
          ),
          article_change: current - previous,
        };
      })
      .sort(
        (a, b) =>
          Math.abs(b.article_change) - Math.abs(a.article_change)
      );
  }, [data.growth, topicNames]);

  const categoryChart = useMemo(() => {
    if (!data.categories.length) return [];

    const sample = data.categories[0];
    const monthKey = findKey(sample, ["month", "current_month", "date"]);
    const categoryKey = findKey(sample, ["category", "topic", "name"]);
    const valueKey = findKey(sample, [
      "article_count",
      "count",
      "percentage",
      "share",
      "category_count",
    ]);

    if (!monthKey || !categoryKey || !valueKey) return [];

    const grouped: Record<string, Row> = {};

    data.categories.forEach((row) => {
      const month = String(row[monthKey] ?? "");
      const category = String(row[categoryKey] ?? "Unknown");
      if (!month) return;
      if (!grouped[month]) grouped[month] = { month };
      grouped[month][category] = getNumber(row[valueKey]) ?? 0;
    });

    return Object.values(grouped).sort((a, b) =>
      String(a.month).localeCompare(String(b.month))
    );
  }, [data.categories]);

  const categoryNames = useMemo(() => {
    if (!categoryChart.length) return [];
    return Object.keys(categoryChart[0]).filter((key) => key !== "month");
  }, [categoryChart]);

  const overview = data.overview || {};

  const topicCount = getNumber(overview.total_topics);
  const totalArticles = getNumber(overview.total_articles);
  const unassigned = getNumber(overview.unassigned_articles);
  const emergingArticles = getNumber(overview.total_emerging_articles);
  const emergingTopics = getNumber(overview.total_emerging_topics);
  const stabilityScore = getNumber(overview.average_topic_stability);
  const emergingPercentage = getNumber(overview.emerging_article_percentage);

  const scrollToSection = (id: string) => {
    setActiveSection(id);
    document.getElementById(id)?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  };

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="loading-mark">
          <Activity size={27} />
        </div>
        <h2>Streamline Engine</h2>
        <p>Loading streaming intelligence data...</p>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Activity size={23} />
          </div>
          <div>
            <strong>STREAMLINE</strong>
            <span>NEWS INTELLIGENCE ENGINE</span>
          </div>
        </div>

        <div className="sidebar-label">WORKSPACE</div>

        <nav className="nav-list">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                className={`nav-item ${
                  activeSection === item.id ? "active" : ""
                }`}
                onClick={() => scrollToSection(item.id)}
              >
                <Icon size={17} />
                <span>{item.label}</span>
                <ChevronRight size={14} />
              </button>
            );
          })}
        </nav>

        <div className="sidebar-bottom">
          <div className="status-dot" />
          <div>
            <strong>Backend connection</strong>
            <span>{error ? "Partial data connection" : "API connected"}</span>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <strong>NEWS STREAM INTELLIGENCE</strong>
            <div className="breadcrumb">Dashboard / Analytics</div>
          </div>

          <div className="topbar-right">
            <div className="dataset-status">
              <span className="status-dot" />
              HISTORICAL STREAM DATA
            </div>
            <button className="refresh-button" onClick={loadData}>
              <RefreshCw size={14} />
              Refresh
            </button>
          </div>
        </header>

        <div className="dashboard">
          {error && <div className="error-banner">{error}</div>}

          {/* OVERVIEW */}
          <section id="overview" className="dashboard-section">
            <div className="hero">
              <div>
                <p className="eyebrow">STREAMING ANALYTICS PLATFORM</p>
                <h1>
                  Understand the <span>News Stream.</span>
                </h1>
                <p className="hero-description">
                  Track evolving topics, discover emerging discussions,
                  monitor semantic changes, and analyze relationships across
                  a chronological collection of news articles.
                </p>
                <div className="hero-meta">
                  <span><Database size={14} /> 2012–2018 dataset</span>
                  <span><Layers3 size={14} /> Monthly processing</span>
                  <span><ShieldCheck size={14} /> Evaluation enabled</span>
                </div>
              </div>

              <div className="hero-graphic">
                <div className="orbit orbit-one" />
                <div className="orbit orbit-two" />
                <div className="orbit orbit-three" />
                <div className="orbit-core">
                  <BrainCircuit size={31} />
                </div>
                <div className="orbit-point point-one" />
                <div className="orbit-point point-two" />
                <div className="orbit-point point-three" />
              </div>
            </div>

            <SectionHeading
              eyebrow="SYSTEM OVERVIEW"
              title="Streaming at a glance"
              description="A summary of the processed news collection and topic discovery results."
            />

            <div className="metrics-grid">
              <MetricCard
                title="Total Articles"
                value={formatNumber(totalArticles)}
                description="Articles processed across all monthly batches"
                icon={FileText}
              />
              <MetricCard
                title="Time Periods"
                value={formatNumber(overview.total_months)}
                description="Chronological monthly batches"
                icon={Clock3}
                color="#8B9CFF"
              />
              <MetricCard
                title="Discovered Topics"
                value={formatNumber(topicCount)}
                description="Initial and emerging topic groups"
                icon={BrainCircuit}
                color="#B58CFF"
              />
              <MetricCard
                title="Emerging Topics"
                value={formatNumber(emergingTopics)}
                description="New topic groups identified"
                icon={TrendingUp}
                color="#F6B85E"
              />
              <MetricCard
                title="Unassigned Articles"
                value={formatNumber(unassigned)}
                description="Articles not assigned to a topic"
                icon={Layers3}
                color="#FF718B"
              />
            </div>

            <div className="metrics-grid metrics-grid-small">
              <MetricCard
                title="Emerging Articles"
                value={formatNumber(emergingArticles)}
                description={`${formatPercent(emergingPercentage)} of all articles`}
                icon={Activity}
              />
              <MetricCard
                title="Average Topic Stability"
                value={
                  stabilityScore === null
                    ? "—"
                    : `${(stabilityScore * 100).toFixed(2)}%`
                }
                description="Average topic-ID overlap score"
                icon={ShieldCheck}
                color="#9BD88B"
              />
              <MetricCard
                title="Evaluation Runtime"
                value={`${formatNumber(overview.evaluation_runtime_seconds)} s`}
                description="Time taken by evaluation calculations"
                icon={Clock3}
                color="#F6B85E"
              />
            </div>

            <div className="two-column">
              <Panel
                title="Topic activity over time"
                description="Monthly article counts for discovered topics."
              >
                {topicActivityChart.length ? (
                  <div className="chart-wrap">
                    <ResponsiveContainer width="100%" height={280}>
                      <LineChart data={topicActivityChart}>
                        <CartesianGrid stroke="#26364e" strokeDasharray="3 3" />
                        <XAxis
                          dataKey="month"
                          stroke="#8998ae"
                          tick={{ fontSize: 9 }}
                          minTickGap={35}
                        />
                        <YAxis stroke="#8998ae" tick={{ fontSize: 10 }} />
                        <Tooltip contentStyle={TOOLTIP_STYLE} />
                        <Legend wrapperStyle={{ fontSize: 10 }} />
                        {topTopics.slice(0, 5).map((topic, index) => (
                          <Line
                            key={topic.id}
                            type="monotone"
                            dataKey={`topic_${topic.id}`}
                            name={topic.name}
                            stroke={COLORS[index % COLORS.length]}
                            dot={false}
                            strokeWidth={2}
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <EmptyState />
                )}
              </Panel>

              <Panel
                title="Largest discovered topics"
                description="Total articles assigned to each topic across the dataset."
              >
                {topTopics.length ? (
                  <div className="chart-wrap">
                    <ResponsiveContainer width="100%" height={280}>
                      <BarChart
                        data={topTopics}
                        layout="vertical"
                        margin={{ left: 5, right: 15 }}
                      >
                        <CartesianGrid stroke="#26364e" strokeDasharray="3 3" />
                        <XAxis type="number" stroke="#8998ae" tick={{ fontSize: 10 }} />
                        <YAxis
                          type="category"
                          dataKey="name"
                          stroke="#8998ae"
                          tick={{ fontSize: 10 }}
                          width={145}
                        />
                        <Tooltip contentStyle={TOOLTIP_STYLE} />
                        <Bar dataKey="count" fill="#2DE2C5" radius={[0, 5, 5, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <EmptyState />
                )}
              </Panel>
            </div>
          </section>

          {/* STREAMING TIMELINE */}
          <section id="timeline" className="dashboard-section">
            <SectionHeading
              eyebrow="CHRONOLOGICAL PROCESSING"
              title="Streaming Timeline"
              description="Explore how the monthly news batches were processed over time."
            />

            <div className="metrics-grid metrics-grid-small">
              <MetricCard
                title="Monthly Batches"
                value={formatNumber(overview.total_months)}
                description="Total chronological batches"
                icon={Clock3}
              />
              <MetricCard
                title="Total Articles"
                value={formatNumber(totalArticles)}
                description="Articles across all batches"
                icon={FileText}
                color="#8B9CFF"
              />
              <MetricCard
                title="Activity Records"
                value={formatNumber(data.activity.length)}
                description="Monthly topic activity observations"
                icon={BarChart3}
                color="#F6B85E"
              />
            </div>

            <Panel
              title="Monthly article volume"
              description="Number of articles processed in each monthly batch."
            >
              {data.performance.length ? (
                <div className="chart-wrap">
                  <ResponsiveContainer width="100%" height={320}>
                    <AreaChart data={data.performance}>
                      <defs>
                        <linearGradient id="articleGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#2DE2C5" stopOpacity={0.35} />
                          <stop offset="95%" stopColor="#2DE2C5" stopOpacity={0.02} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid stroke="#26364e" strokeDasharray="3 3" />
                      <XAxis dataKey="month" stroke="#8998ae" tick={{ fontSize: 9 }} minTickGap={35} />
                      <YAxis stroke="#8998ae" tick={{ fontSize: 10 }} />
                      <Tooltip contentStyle={TOOLTIP_STYLE} />
                      <Area
                        type="monotone"
                        dataKey="article_count"
                        name="Articles"
                        stroke="#2DE2C5"
                        fill="url(#articleGradient)"
                        strokeWidth={2}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <EmptyState />
              )}
            </Panel>

            <div className="two-column">
              <Panel
                title="Processing performance"
                description="Monthly batch preparation and topic-assignment time and throughput."
              >
                <DataTable
                  rows={data.performance}
                  columns={[
                    "month",
                    "article_count",
                    "processing_seconds",
                    "articles_per_second",
                  ]}
                    limit={data.performance.length}
                />
              </Panel>

              <Panel
                title="Latest monthly topic activity"
                description="Topics with the highest article counts in the latest available month."
              >
                <DataTable
                  rows={latestActivity}
                  columns={["month", "topic_name", "topic_id", "article_count"]}
                />
              </Panel>
            </div>

            <div className="method-note">
              Processing time measures monthly batch preparation and topic
              assignment; report and file-writing time is excluded.
            </div>
          </section>

          {/* TOPIC INTELLIGENCE */}
          <section id="topics" className="dashboard-section">
            <SectionHeading
              eyebrow="DISCOVERY AND TRACKING"
              title="Topic Intelligence"
              description="Inspect topic activity, growth, stability, and emerging topic groups."
            />

            <div className="metrics-grid metrics-grid-small">
              <MetricCard
                title="Total Topics"
                value={formatNumber(topicCount)}
                description="Topics discovered by the engine"
                icon={BrainCircuit}
              />
              <MetricCard
                title="Emerging Topics"
                value={formatNumber(emergingTopics)}
                description="Emerging topic groups"
                icon={TrendingUp}
                color="#F6B85E"
              />
              <MetricCard
                title="Emerging Articles"
                value={formatNumber(emergingArticles)}
                description="Articles assigned to emerging topics"
                icon={FileText}
                color="#8B9CFF"
              />
            </div>

            <Panel
              title="Topic cards"
              description="Topics ranked by total article volume. Names are generated from each topic's category and headline terms."
            >
              {topTopics.length ? (
                <div className="topic-grid">
                  {topTopics.map((topic, index) => (
                    <div className="topic-card" key={topic.id}>
                      <div className="topic-card-top">
                        <span className="topic-number">
                          TOPIC {topic.id}
                        </span>
                        <span
                          className="topic-dot"
                          style={{ background: COLORS[index % COLORS.length] }}
                        />
                      </div>
                      <div className="topic-name">{topic.name}</div>
                      <div className="metric-value">
                        {formatNumber(topic.count)}
                      </div>
                      <p>Articles assigned</p>
                    </div>
                  ))}
                </div>
              ) : (
                <EmptyState />
              )}
            </Panel>

            <div
              className={`two-column growth-chart-grid ${
                growthChartExpanded ? "expanded" : ""
              }`}
            >
              <Panel
                title="Monthly topic growth"
                description="Monthly percentage change for the five highest-volume topics; blank means no prior count. Use the time window to zoom."
              >
                {visibleTopicGrowth.length ? (
                  <>
                    <div className="chart-toolbar">
                      <label className="chart-control">
                        <span>Time window</span>
                        <select
                          value={growthWindow}
                          onChange={(event) =>
                            setGrowthWindow(event.target.value)
                          }
                        >
                          <option value="12">Last 12 months</option>
                          <option value="24">Last 24 months</option>
                          <option value="36">Last 36 months</option>
                          <option value="60">Last 60 months</option>
                          <option value="all">All months</option>
                        </select>
                      </label>
                      <button
                        type="button"
                        className="chart-size-button"
                        aria-expanded={growthChartExpanded}
                        onClick={() =>
                          setGrowthChartExpanded((expanded) => !expanded)
                        }
                      >
                        {growthChartExpanded ? "Compact chart" : "Expand chart"}
                      </button>
                    </div>
                    <div className="chart-wrap">
                      <ResponsiveContainer
                        width="100%"
                        height={growthChartExpanded ? 440 : 300}
                      >
                        <LineChart data={visibleTopicGrowth}>
                          <CartesianGrid
                            stroke="#26364e"
                            strokeDasharray="3 3"
                          />
                          <XAxis
                            dataKey="month"
                            stroke="#8998ae"
                            tick={{ fontSize: 9 }}
                            minTickGap={35}
                          />
                          <YAxis
                            stroke="#8998ae"
                            tick={{ fontSize: 10 }}
                            tickFormatter={(value) => `${value}%`}
                          />
                          <Tooltip contentStyle={TOOLTIP_STYLE} />
                          <Legend wrapperStyle={{ fontSize: 10 }} />
                          {topTopics.slice(0, 5).map((topic, index) => (
                            <Line
                              key={topic.id}
                              type="monotone"
                              dataKey={`topic_${topic.id}`}
                              name={topic.name}
                              stroke={COLORS[index % COLORS.length]}
                              dot={false}
                              strokeWidth={2}
                            />
                          ))}
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </>
                ) : (
                  <EmptyState />
                )}
              </Panel>

              <Panel
                title="Topic stability"
                description="Jaccard stability between consecutive monthly topic sets."
              >
                {data.stability.length ? (
                  <div className="chart-wrap">
                    <ResponsiveContainer width="100%" height={280}>
                      <LineChart data={data.stability}>
                        <CartesianGrid stroke="#26364e" strokeDasharray="3 3" />
                        <XAxis dataKey="current_month" stroke="#8998ae" tick={{ fontSize: 9 }} minTickGap={35} />
                        <YAxis
                          domain={[0, 1]}
                          stroke="#8998ae"
                          tick={{ fontSize: 10 }}
                        />
                        <Tooltip contentStyle={TOOLTIP_STYLE} />
                        <Line
                          type="monotone"
                          dataKey="topic_jaccard_stability"
                          name="Topic Jaccard Stability"
                          stroke="#2DE2C5"
                          dot={false}
                          strokeWidth={2}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <EmptyState />
                )}
              </Panel>
            </div>

            <Panel
              title="Emerging topic records"
              description="Records returned by the emerging-topic evaluation endpoint."
            >
              <DataTable rows={data.emerging} />
            </Panel>

            <Panel
              title="Latest monthly topic growth"
              description="Latest month, sorted by absolute article-count change."
            >
              <DataTable
                rows={latestTopicGrowth}
                columns={[
                  "month",
                  "topic_name",
                  "topic_id",
                  "previous_articles",
                  "current_articles",
                  "article_change",
                  "growth_percentage",
                ]}
                limit={latestTopicGrowth.length}
              />
            </Panel>
          </section>

          {/* TREND ANALYSIS */}
          <section id="trends" className="dashboard-section">
            <SectionHeading
              eyebrow="CATEGORY MOVEMENT"
              title="Trend Analysis"
              description="Monitor changes in category activity throughout the historical news stream."
            />

            <Panel
              title="Category activity over time"
              description="Category-level trends from the category analysis output."
            >
              {categoryChart.length && categoryNames.length ? (
                <div className="chart-wrap">
                  <ResponsiveContainer width="100%" height={330}>
                    <LineChart data={categoryChart}>
                      <CartesianGrid stroke="#26364e" strokeDasharray="3 3" />
                      <XAxis dataKey="month" stroke="#8998ae" tick={{ fontSize: 9 }} minTickGap={35} />
                      <YAxis stroke="#8998ae" tick={{ fontSize: 10 }} />
                      <Tooltip contentStyle={TOOLTIP_STYLE} />
                      <Legend wrapperStyle={{ fontSize: 10 }} />
                      {categoryNames.slice(0, 8).map((category, index) => (
                        <Line
                          key={category}
                          type="monotone"
                          dataKey={category}
                          name={category}
                          stroke={COLORS[index % COLORS.length]}
                          dot={false}
                          strokeWidth={2}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <EmptyState message="Category trend records are available, but their field structure does not match the expected month/category/value format. See the raw records below." />
              )}
            </Panel>

            <Panel
              title="Category trend records"
              description="Source records returned by the category trends endpoint."
            >
              <DataTable rows={data.categories} />
            </Panel>
          </section>

          {/* SEMANTIC DRIFT */}
          <section id="drift" className="dashboard-section">
            <SectionHeading
              eyebrow="MEANING OVER TIME"
              title="Semantic Drift"
              description="Measure how the semantic representation of news changes between consecutive months."
            />

            <div className="metrics-grid metrics-grid-small">
              <MetricCard
                title="Monthly Comparisons"
                value={formatNumber(data.drift.length)}
                description="Records in semantic drift analysis"
                icon={GitBranch}
              />
              <MetricCard
                title="Latest Semantic Drift"
                value={
                  data.drift.length
                    ? formatNumber(data.drift[data.drift.length - 1].semantic_drift)
                    : "—"
                }
                description="Latest available semantic drift score"
                icon={Activity}
                color="#F6B85E"
              />
              <MetricCard
                title="Latest Similarity"
                value={
                  data.drift.length
                    ? formatNumber(data.drift[data.drift.length - 1].cosine_similarity)
                    : "—"
                }
                description="Cosine similarity with previous month"
                icon={ShieldCheck}
                color="#9BD88B"
              />
            </div>

            <Panel
              title="Semantic similarity and drift"
              description="Cosine similarity and semantic drift across monthly comparisons."
            >
              {data.drift.length ? (
                <div className="chart-wrap">
                  <ResponsiveContainer width="100%" height={320}>
                    <LineChart data={data.drift}>
                      <CartesianGrid stroke="#26364e" strokeDasharray="3 3" />
                      <XAxis dataKey="month" stroke="#8998ae" tick={{ fontSize: 9 }} minTickGap={35} />
                      <YAxis domain={[0, 1]} stroke="#8998ae" tick={{ fontSize: 10 }} />
                      <Tooltip contentStyle={TOOLTIP_STYLE} />
                      <Legend wrapperStyle={{ fontSize: 10 }} />
                      <Line
                        type="monotone"
                        dataKey="cosine_similarity"
                        name="Cosine Similarity"
                        stroke="#2DE2C5"
                        dot={false}
                        strokeWidth={2}
                        connectNulls
                      />
                      <Line
                        type="monotone"
                        dataKey="semantic_drift"
                        name="Semantic Drift"
                        stroke="#FF718B"
                        dot={false}
                        strokeWidth={2}
                        connectNulls
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <EmptyState />
              )}
            </Panel>

            <Panel
              title="Distribution drift"
              description="Monthly distribution drift values."
            >
              {data.drift.length ? (
                <div className="chart-wrap">
                  <ResponsiveContainer width="100%" height={260}>
                    <AreaChart data={data.drift}>
                      <CartesianGrid stroke="#26364e" strokeDasharray="3 3" />
                      <XAxis dataKey="month" stroke="#8998ae" tick={{ fontSize: 9 }} minTickGap={35} />
                      <YAxis stroke="#8998ae" tick={{ fontSize: 10 }} />
                      <Tooltip contentStyle={TOOLTIP_STYLE} />
                      <Area
                        type="monotone"
                        dataKey="distribution_drift"
                        name="Distribution Drift"
                        stroke="#8B9CFF"
                        fill="#8B9CFF"
                        fillOpacity={0.12}
                        connectNulls
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <EmptyState />
              )}
            </Panel>

            <div className="method-note">
              Semantic drift and distribution drift are separate measurements.
              The first is based on changes in semantic similarity; the second
              tracks changes in the distribution of the analyzed data.
            </div>
          </section>

          {/* ARTICLE NETWORK */}
          <section id="network" className="dashboard-section">
            <SectionHeading
              eyebrow="RELATIONSHIP AND INFLUENCE"
              title="Article Network"
              description="Explore influential articles using PageRank and network degree."
            />

            <div className="metrics-grid metrics-grid-small">
              <MetricCard
                title="Network Articles"
                value={formatNumber(data.network.length)}
                description="Article records returned by the network endpoint"
                icon={Network}
              />
              <MetricCard
                title="Influential Records"
                value={formatNumber(data.influential.length)}
                description="Records in the influential articles output"
                icon={TrendingUp}
                color="#F6B85E"
              />
              <MetricCard
                title="Network Edges"
                value={
                  data.networkEdges === null
                    ? "—"
                    : formatNumber(data.networkEdges)
                }
                description="Edges in the article similarity graph"
                icon={GitBranch}
                color="#8B9CFF"
              />
            </div>

            <Panel
              title="Top articles by PageRank"
              description="Articles with the highest PageRank scores in the network output."
            >
              <DataTable
                rows={[...data.network].sort(
                  (a, b) =>
                    (getNumber(b.pagerank) || 0) -
                    (getNumber(a.pagerank) || 0)
                )}
                columns={[
                  "id",
                  "date",
                  "category",
                  "pagerank",
                  "degree",
                  "text_preview",
                ]}
                limit={15}
              />
            </Panel>

            <Panel
              title="Influential articles"
              description="Precomputed influential article records."
            >
              <DataTable rows={data.influential} limit={15} />
            </Panel>
          </section>

          {/* EVALUATION */}
          <section id="evaluation" className="dashboard-section">
            <SectionHeading
              eyebrow="SYSTEM PERFORMANCE"
              title="Evaluation"
              description="Review the quantitative results generated by the streaming evaluation pipeline."
            />

            <div className="metrics-grid">
              <MetricCard
                title="Total Articles"
                value={formatNumber(totalArticles)}
                description="Total processed news articles"
                icon={FileText}
              />
              <MetricCard
                title="Total Topics"
                value={formatNumber(topicCount)}
                description="Discovered topic groups"
                icon={BrainCircuit}
                color="#8B9CFF"
              />
              <MetricCard
                title="Topic Stability"
                value={
                  stabilityScore === null
                    ? "—"
                    : `${(stabilityScore * 100).toFixed(2)}%`
                }
                description="Average topic-ID overlap score"
                icon={ShieldCheck}
                color="#9BD88B"
              />
              <MetricCard
                title="Emerging Article Share"
                value={formatPercent(emergingPercentage)}
                description="Percentage of articles in emerging topics"
                icon={TrendingUp}
                color="#F6B85E"
              />
              <MetricCard
                title="Unassigned Articles"
                value={formatNumber(unassigned)}
                description="Articles without a topic assignment"
                icon={Layers3}
                color="#FF718B"
              />
            </div>

            <div className="two-column">
              <Panel
                title="Topic stability evaluation"
                description="Monthly Jaccard stability values."
              >
                {data.stability.length ? (
                  <div className="chart-wrap">
                    <ResponsiveContainer width="100%" height={270}>
                      <AreaChart data={data.stability}>
                        <CartesianGrid stroke="#26364e" strokeDasharray="3 3" />
                        <XAxis dataKey="current_month" stroke="#8998ae" tick={{ fontSize: 9 }} minTickGap={35} />
                        <YAxis domain={[0, 1]} stroke="#8998ae" tick={{ fontSize: 10 }} />
                        <Tooltip contentStyle={TOOLTIP_STYLE} />
                        <Area
                          type="monotone"
                          dataKey="topic_jaccard_stability"
                          name="Jaccard Stability"
                          stroke="#2DE2C5"
                          fill="#2DE2C5"
                          fillOpacity={0.12}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <EmptyState />
                )}
              </Panel>

              <Panel
                title="Evaluation summary"
                description="Overall evaluation metrics returned by the backend."
              >
                <DataTable
                  rows={Object.entries(overview).map(([metric, value]) => ({
                    metric,
                    value: value ?? "—",
                  }))}
                  columns={["metric", "value"]}
                  limit={20}
                />
              </Panel>
            </div>

            <Panel
              title="Processing performance records"
              description="Monthly article counts and processing measurements."
            >
              <DataTable
                rows={data.performance}
                limit={data.performance.length}
              />
            </Panel>

            <div className="method-note">
              Interpretation note: topic stability here represents topic-ID
              overlap between consecutive periods. It should not be treated
              as an independent measure of semantic topic coherence.
              Emerging-topic counts describe detected groups, not verified
              precision or recall.
            </div>
          </section>

          <footer className="footer">
            <div className="footer-brand">
              <Activity size={16} />
              STREAMLINE ENGINE
            </div>
            <span>Historical news stream analytics · 2012–2018</span>
            <span>Last updated: {lastUpdated || "—"}</span>
          </footer>
        </div>
      </main>
    </div>
  );
}
