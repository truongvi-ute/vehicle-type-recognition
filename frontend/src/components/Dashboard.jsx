import { useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  ArrowRight,
  BarChart3,
  CheckCircle2,
  Clock3,
  Database,
  GitCompareArrows,
  Grid3X3,
  Info,
  LineChart,
  Loader2,
  RefreshCw,
  Settings,
  ShieldCheck,
  TableProperties,
  TrendingDown,
  TrendingUp,
} from "lucide-react";

import {
  fetchDatasetProfile,
  fetchExperiment,
  fetchExperimentCatalog,
  fetchExperimentComparison,
} from "../api/metricsApi";

const NUMBER = new Intl.NumberFormat("vi-VN");
const BUCKET_LABELS = {
  normal: "Normal",
  rain: "Rain",
  sun: "Sun",
  night: "Night",
  gaussian: "Gaussian blur",
  motion: "Motion blur",
  unsharp: "Unsharp mask",
  gaussian_blur: "Gaussian blur",
  motion_blur: "Motion blur",
  unsharp_mask: "Unsharp mask",
};
const BUCKET_COLORS = ["#2563eb", "#0f766e", "#d97706", "#7c3aed"];

function formatNumber(value) {
  return Number.isFinite(value) ? NUMBER.format(value) : "—";
}

function formatPercent(value, digits = 2) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "—";
}

function formatDelta(value) {
  if (!Number.isFinite(value)) return "—";
  const points = value * 100;
  return `${points > 0 ? "+" : ""}${points.toFixed(2)} điểm %`;
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return "Không được ghi nhận";
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.round((seconds % 3600) / 60);
  return hours ? `${hours} giờ ${minutes} phút` : `${minutes} phút`;
}

function getItem(items, id) {
  return items?.find((item) => item.id === id);
}

function SectionHeading({ icon: Icon, title, description, action }) {
  return (
    <div className="exp-section-heading">
      <div>
        <h3><Icon size={18} />{title}</h3>
        {description ? <p>{description}</p> : null}
      </div>
      {action}
    </div>
  );
}

function LoadingState({ label = "Đang đọc dữ liệu..." }) {
  return <div className="exp-loading"><Loader2 className="spin" size={20} />{label}</div>;
}

function ErrorState({ message, onRetry }) {
  return (
    <div className="exp-message exp-message-error">
      <AlertCircle size={20} />
      <div><strong>Không thể hiển thị dữ liệu</strong><span>{message}</span></div>
      {onRetry ? <button type="button" onClick={onRetry}><RefreshCw size={16} />Thử lại</button> : null}
    </div>
  );
}

function DatasetOverview({ profile }) {
  if (!profile) return null;
  const buckets = Object.entries(profile.bucket_totals || {});
  const maxBucket = Math.max(...buckets.map(([, value]) => value), 1);

  return (
    <section className="exp-section">
      <SectionHeading
        icon={Database}
        title="Dữ liệu đầu vào"
        description="Các số lượng dưới đây được đọc trực tiếp từ báo cáo data preparation."
      />
      <div className="exp-data-stats">
        <div><span>Ảnh raw</span><strong>{formatNumber(profile.raw_total)}</strong></div>
        <div><span>Train nguồn</span><strong>{formatNumber(profile.split_totals?.train)}</strong></div>
        <div><span>Train sau cân bằng</span><strong>{formatNumber(profile.train_total)}</strong></div>
        <div><span>Validation unseen</span><strong>{formatNumber(profile.split_totals?.valid_unseen)}</strong></div>
        <div><span>Test cố định</span><strong>{formatNumber(profile.split_totals?.test)}</strong></div>
      </div>

      <div className="exp-bucket-list" aria-label="Phân bố augmentation">
        {buckets.map(([name, value], index) => (
          <div className="exp-bucket-row" key={name}>
            <span>{BUCKET_LABELS[name] || name}</span>
            <div className="exp-track"><i style={{ width: `${(value / maxBucket) * 100}%`, background: BUCKET_COLORS[index % BUCKET_COLORS.length] }} /></div>
            <strong>{formatNumber(value)}</strong>
            <small>{formatPercent(value / profile.train_total, 1)}</small>
          </div>
        ))}
      </div>

      <details className="exp-details">
        <summary>Chi tiết phân bố theo lớp</summary>
        <div className="exp-table-scroll">
          <table className="exp-table">
            <thead><tr><th>Lớp</th><th>Raw</th><th>Train</th><th>Validation</th><th>Test</th></tr></thead>
            <tbody>
              {profile.class_distribution?.map((row) => (
                <tr key={row.class_name}>
                  <th>{row.class_name}</th>
                  <td>{formatNumber(row.raw)}</td>
                  <td>{formatNumber(row.train)}</td>
                  <td>{formatNumber(row.valid_unseen)}</td>
                  <td>{formatNumber(row.test)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </section>
  );
}

function TrainingChart({ history }) {
  const [metric, setMetric] = useState("loss");
  const width = 820;
  const height = 270;
  const padding = { left: 52, right: 24, top: 22, bottom: 38 };
  const series = metric === "loss"
    ? [
        { key: "train_loss", label: "Train loss", color: "#2563eb" },
        { key: "valid_unseen_loss", label: "Validation loss", color: "#d97706" },
      ]
    : [{ key: "valid_unseen_acc", label: "Validation accuracy", color: "#0f766e" }];
  const values = history.flatMap((row) => series.map((item) => row[item.key])).filter(Number.isFinite);
  const dataMin = Math.min(...values);
  const dataMax = Math.max(...values);
  const spread = Math.max(dataMax - dataMin, 0.01);
  const minY = metric === "accuracy" ? Math.max(0, dataMin - spread * 0.2) : Math.max(0, dataMin - spread * 0.15);
  const maxY = Math.min(metric === "accuracy" ? 1 : Infinity, dataMax + spread * 0.15);
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  const x = (index) => padding.left + (index / Math.max(history.length - 1, 1)) * chartWidth;
  const y = (value) => padding.top + (1 - (value - minY) / Math.max(maxY - minY, 0.001)) * chartHeight;

  return (
    <section className="exp-section">
      <SectionHeading
        icon={LineChart}
        title="Diễn biến huấn luyện"
        description="Đường cong được dựng từ toàn bộ lịch sử epoch đã lưu."
        action={(
          <div className="exp-segmented">
            <button type="button" className={metric === "loss" ? "active" : ""} onClick={() => setMetric("loss")}>Loss</button>
            <button type="button" className={metric === "accuracy" ? "active" : ""} onClick={() => setMetric("accuracy")}>Accuracy</button>
          </div>
        )}
      />
      <div className="exp-chart-wrap">
        <svg className="exp-line-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Đường cong ${metric}`}>
          {[0, 0.25, 0.5, 0.75, 1].map((fraction) => {
            const chartY = padding.top + chartHeight * fraction;
            const value = maxY - (maxY - minY) * fraction;
            return <g key={fraction}><line x1={padding.left} x2={width - padding.right} y1={chartY} y2={chartY} className="exp-grid-line" /><text x={padding.left - 9} y={chartY + 4} textAnchor="end">{metric === "accuracy" ? `${(value * 100).toFixed(1)}%` : value.toFixed(2)}</text></g>;
          })}
          {series.map((item) => {
            const points = history.map((row, index) => `${x(index)},${y(row[item.key])}`).join(" ");
            return <polyline key={item.key} points={points} fill="none" stroke={item.color} strokeWidth="3" strokeLinejoin="round" strokeLinecap="round" />;
          })}
          {history.map((row, index) => (
            <text key={row.epoch} x={x(index)} y={height - 12} textAnchor="middle">{row.epoch}</text>
          ))}
        </svg>
      </div>
      <div className="exp-legend">
        {series.map((item) => <span key={item.key}><i style={{ background: item.color }} />{item.label}</span>)}
        <span className="exp-axis-note">Epoch</span>
      </div>
    </section>
  );
}

function ClassPerformance({ rows }) {
  const sorted = [...rows].sort((a, b) => b.f1 - a.f1);
  return (
    <section className="exp-section">
      <SectionHeading icon={BarChart3} title="Hiệu quả theo lớp" description="F1-score giúp nhìn rõ lớp mạnh và lớp còn yếu, không bị che bởi lớp có nhiều ảnh." />
      <div className="exp-class-bars">
        {sorted.map((row) => (
          <div className="exp-class-row" key={row.class_name}>
            <span>{row.class_name}</span>
            <div className="exp-track"><i style={{ width: `${row.f1 * 100}%` }} /></div>
            <strong>{formatPercent(row.f1, 1)}</strong>
            <small>{formatNumber(row.support)} ảnh</small>
          </div>
        ))}
      </div>
    </section>
  );
}

function ResultUnavailable({ datasetProfile }) {
  return (
    <>
      <DatasetOverview profile={datasetProfile} />
      <div className="exp-message exp-message-info">
        <Info size={21} />
        <div>
          <strong>Chưa có kết quả huấn luyện cho tổ hợp này</strong>
          <span>Dashboard chỉ hiển thị phần data preparation đã có thật. Khi output đánh giá được đăng ký, các chỉ số model sẽ tự xuất hiện.</span>
        </div>
      </div>
    </>
  );
}

function OverviewView({ experiment, datasetProfile }) {
  if (!experiment) return <ResultUnavailable datasetProfile={datasetProfile} />;
  const { evaluation, training } = experiment;
  return (
    <div className="exp-view-stack">
      <div className="exp-kpi-grid">
        <div className="exp-kpi"><span>Test accuracy</span><strong>{formatPercent(evaluation.test.accuracy)}</strong><small>Trên {formatNumber(evaluation.test.samples)} ảnh</small></div>
        <div className="exp-kpi"><span>Macro F1</span><strong>{formatPercent(evaluation.test.macro_f1)}</strong><small>Cân bằng giữa 10 lớp</small></div>
        <div className="exp-kpi"><span>Epoch tốt nhất</span><strong>{formatNumber(training.best_epoch)}</strong><small>{formatNumber(training.completed_epochs)} epoch đã chạy</small></div>
        <div className="exp-kpi"><span>Validation accuracy</span><strong>{formatPercent(evaluation.valid_unseen.accuracy)}</strong><small>{formatNumber(evaluation.valid_unseen.samples)} ảnh unseen</small></div>
      </div>
      <DatasetOverview profile={datasetProfile} />
      <TrainingChart history={experiment.history} />
      <ClassPerformance rows={evaluation.class_metrics} />
      <section className="exp-section">
        <SectionHeading icon={ShieldCheck} title="Nhận định từ kết quả" description="Nhận định được tính trực tiếp từ F1, validation gap và ma trận nhầm lẫn." />
        <div className="exp-insights">
          {experiment.insights.map((insight) => <div key={insight.kind}><CheckCircle2 size={18} /><span>{insight.message}</span></div>)}
        </div>
        <details className="exp-details">
          <summary>Nguồn dữ liệu dùng cho màn hình này</summary>
          <dl className="exp-source-list">
            {Object.entries(experiment.sources).map(([name, path]) => <div key={name}><dt>{name}</dt><dd>{path}</dd></div>)}
          </dl>
        </details>
      </section>
    </div>
  );
}

function ConfusionMatrix({ evaluation }) {
  const [mode, setMode] = useState("count");
  const matrix = evaluation.confusion_matrix;

  // Maximums for diagonal (correct)
  const maxDiagCount = Math.max(...matrix.map((row, idx) => row[idx]), 1);
  const maxDiagRate = Math.max(
    ...matrix.map((row, idx) => {
      const rTotal = row.reduce((s, v) => s + v, 0);
      return rTotal ? row[idx] / rTotal : 0;
    }),
    0.001
  );

  // Maximums for off-diagonal (confusion)
  const maxConfusionCount = Math.max(
    ...matrix.flatMap((row, rIdx) => row.filter((_, cIdx) => rIdx !== cIdx)),
    1
  );
  
  const offDiagonalRates = matrix.flatMap((row, rIdx) => {
    const rTotal = row.reduce((s, v) => s + v, 0);
    return row.map((c, cIdx) => (rIdx !== cIdx && rTotal) ? c / rTotal : 0);
  });
  const maxConfusionRate = Math.max(...offDiagonalRates, 0.001);

  return (
    <section className="exp-section">
      <SectionHeading
        icon={Grid3X3}
        title="Ma trận nhầm lẫn"
        description="Hàng là nhãn thật, cột là nhãn dự đoán."
        action={(
          <div className="exp-segmented">
            <button type="button" className={mode === "count" ? "active" : ""} onClick={() => setMode("count")}>Số ảnh</button>
            <button type="button" className={mode === "rate" ? "active" : ""} onClick={() => setMode("rate")}>Theo hàng</button>
          </div>
        )}
      />
      <div className="exp-table-scroll">
        <table className="exp-matrix">
          <thead><tr><th>Thật / đoán</th>{evaluation.class_names.map((name) => <th key={name}>{name}</th>)}</tr></thead>
          <tbody>
            {matrix.map((row, rowIndex) => {
              const rowTotal = row.reduce((sum, value) => sum + value, 0);
              return (
                <tr key={evaluation.class_names[rowIndex]}>
                  <th>{evaluation.class_names[rowIndex]}</th>
                  {row.map((count, colIndex) => {
                    const rate = rowTotal ? count / rowTotal : 0;
                    const correct = rowIndex === colIndex;
                    
                    let intensity;
                    if (correct) {
                      intensity = mode === "count" ? count / maxDiagCount : rate / maxDiagRate;
                    } else {
                      intensity = mode === "count" ? count / maxConfusionCount : rate / maxConfusionRate;
                    }
                    
                    intensity = Math.min(Math.max(intensity, 0), 1);
                    
                    const bgColor = correct
                      ? `rgba(37, 99, 235, ${count > 0 ? 0.08 + intensity * 0.72 : 0})`
                      : `rgba(220, 38, 38, ${count > 0 ? 0.05 + intensity * 0.65 : 0})`;
                      
                    const textColor = (intensity > 0.5 && count > 0) ? "#fff" : undefined;

                    return (
                      <td 
                        key={`${rowIndex}-${colIndex}`} 
                        style={{ backgroundColor: bgColor, color: textColor }}
                      >
                        {mode === "count" ? formatNumber(count) : formatPercent(rate, 1)}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function AnalysisView({ experiment }) {
  const [sortKey, setSortKey] = useState("f1");
  if (!experiment) return <div className="exp-message exp-message-info"><Info size={21} /><div><strong>Chưa có dữ liệu phân tích model</strong><span>Hãy chọn một tổ hợp đã huấn luyện.</span></div></div>;
  const evaluation = experiment.evaluation;
  const rows = [...evaluation.class_metrics].sort((a, b) => b[sortKey] - a[sortKey]);
  return (
    <div className="exp-view-stack">
      <section className="exp-section">
        <SectionHeading
          icon={TableProperties}
          title="Chỉ số từng lớp"
          description="Precision, recall và F1 được lấy từ classification report đã kiểm chứng với test set."
          action={(
            <label className="exp-inline-select">Sắp xếp
              <select value={sortKey} onChange={(event) => setSortKey(event.target.value)}>
                <option value="f1">F1-score</option><option value="precision">Precision</option><option value="recall">Recall</option><option value="support">Số mẫu</option>
              </select>
            </label>
          )}
        />
        <div className="exp-table-scroll">
          <table className="exp-table">
            <thead><tr><th>Lớp</th><th>Precision</th><th>Recall</th><th>F1-score</th><th>Test support</th></tr></thead>
            <tbody>{rows.map((row) => <tr key={row.class_name}><th>{row.class_name}</th><td>{formatPercent(row.precision)}</td><td>{formatPercent(row.recall)}</td><td><strong>{formatPercent(row.f1)}</strong></td><td>{formatNumber(row.support)}</td></tr>)}</tbody>
          </table>
        </div>
      </section>
      <ConfusionMatrix evaluation={evaluation} />
      <section className="exp-section">
        <SectionHeading icon={AlertCircle} title="Các cặp nhầm nhiều nhất" description="Tỷ lệ được tính trên tổng số ảnh của nhãn thật tương ứng." />
        <div className="exp-error-list">
          {evaluation.top_errors.map((error, index) => (
            <div key={`${error.true_class}-${error.predicted_class}`}>
              <span className="exp-rank">{index + 1}</span>
              <strong>{error.true_class}</strong><ArrowRight size={16} /><strong>{error.predicted_class}</strong>
              <span>{formatNumber(error.count)} ảnh</span><small>{formatPercent(error.rate, 1)} của lớp thật</small>
            </div>
          ))}
        </div>
      </section>
      <section className="exp-section">
        <SectionHeading icon={Clock3} title="Tóm tắt quá trình train" />
        <dl className="exp-training-summary">
          <div><dt>Tiêu chí chọn best</dt><dd>{experiment.training.selection_metric === "maximum_valid_unseen_accuracy" ? "Validation accuracy cao nhất" : "Validation loss thấp nhất"}</dd></div>
          <div><dt>Epoch tốt nhất</dt><dd>{formatNumber(experiment.training.best_epoch)}</dd></div>
          <div><dt>Validation loss thấp nhất</dt><dd>{experiment.training.minimum_valid_loss?.toFixed(6) ?? "—"} tại epoch {formatNumber(experiment.training.minimum_valid_loss_epoch)}</dd></div>
          <div><dt>Validation accuracy cao nhất</dt><dd>{formatPercent(experiment.training.maximum_valid_accuracy)} tại epoch {formatNumber(experiment.training.maximum_valid_accuracy_epoch)}</dd></div>
          <div><dt>Tổng thời gian được ghi nhận</dt><dd>{formatDuration(experiment.training.total_elapsed_s)}</dd></div>
        </dl>
      </section>
    </div>
  );
}

function ExperimentLabel({ experiment }) {
  return <span><strong>{experiment.model.label}</strong><small>{experiment.dataset.label} · {experiment.version.label}</small></span>;
}

function CompareView({ catalog }) {
  const available = useMemo(() => catalog.combinations.filter((item) => item.available), [catalog]);
  const [firstId, setFirstId] = useState(available[0]?.id || "");
  const [secondId, setSecondId] = useState(available[1]?.id || available[0]?.id || "");
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const labelFor = (id) => {
    const combination = available.find((item) => item.id === id);
    if (!combination) return id;
    return `${getItem(catalog.models, combination.model)?.label} · ${getItem(catalog.datasets, combination.dataset)?.label} · ${getItem(catalog.versions, combination.version)?.label}`;
  };

  useEffect(() => {
    let cancelled = false;
    if (!firstId || !secondId || firstId === secondId) {
      setComparison(null);
      return undefined;
    }
    setLoading(true);
    setError("");
    fetchExperimentComparison(firstId, secondId)
      .then((data) => { if (!cancelled) setComparison(data); })
      .catch((requestError) => { if (!cancelled) setError(requestError.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [firstId, secondId]);

  return (
    <div className="exp-view-stack">
      <section className="exp-compare-picker">
        <label>Thí nghiệm A<select value={firstId} onChange={(event) => setFirstId(event.target.value)}>{available.map((item) => <option key={item.id} value={item.id}>{labelFor(item.id)}</option>)}</select></label>
        <GitCompareArrows size={22} />
        <label>Thí nghiệm B<select value={secondId} onChange={(event) => setSecondId(event.target.value)}>{available.map((item) => <option key={item.id} value={item.id}>{labelFor(item.id)}</option>)}</select></label>
      </section>
      {firstId === secondId ? <div className="exp-message exp-message-info"><Info size={21} /><div><strong>Chọn hai thí nghiệm khác nhau</strong><span>Chỉ các run có output đánh giá thật mới xuất hiện trong danh sách.</span></div></div> : null}
      {loading ? <LoadingState label="Đang đối chiếu hai kết quả..." /> : null}
      {error ? <ErrorState message={error} /> : null}
      {comparison && !loading ? (
        <>
          {comparison.warnings.map((warning) => <div className="exp-message exp-message-warning" key={warning}><AlertCircle size={21} /><div><strong>Lưu ý khi diễn giải</strong><span>{warning}</span></div></div>)}
          <section className="exp-section">
            <SectionHeading icon={GitCompareArrows} title="Chênh lệch tổng thể" description="Delta được tính B trừ A; số dương nghĩa là B cao hơn." />
            <div className="exp-compare-heads"><ExperimentLabel experiment={comparison.first} /><ExperimentLabel experiment={comparison.second} /></div>
            <div className="exp-compare-metrics">
              {[
                ["Accuracy", "accuracy"], ["Macro F1", "macro_f1"], ["Weighted F1", "weighted_f1"],
              ].map(([label, key]) => {
                const delta = comparison.deltas[key];
                return <div key={key}><span>{label}</span><strong>{formatPercent(comparison.first.evaluation.test[key])}</strong><ArrowRight size={16} /><strong>{formatPercent(comparison.second.evaluation.test[key])}</strong><em className={delta >= 0 ? "positive" : "negative"}>{delta >= 0 ? <TrendingUp size={15} /> : <TrendingDown size={15} />}{formatDelta(delta)}</em></div>;
              })}
            </div>
          </section>
          <section className="exp-section">
            <SectionHeading icon={BarChart3} title="Thay đổi F1 theo lớp" description="Sắp xếp theo độ lớn chênh lệch để thấy lớp nào thay đổi nhiều nhất." />
            <div className="exp-delta-list">
              {[...comparison.class_deltas].sort((a, b) => Math.abs(b.delta_f1) - Math.abs(a.delta_f1)).map((row) => (
                <div key={row.class_name}><strong>{row.class_name}</strong><span>{formatPercent(row.first_f1, 1)}</span><ArrowRight size={15} /><span>{formatPercent(row.second_f1, 1)}</span><em className={row.delta_f1 >= 0 ? "positive" : "negative"}>{formatDelta(row.delta_f1)}</em></div>
              ))}
            </div>
          </section>
        </>
      ) : null}
    </div>
  );
}

function ConfigView({ experiment }) {
  if (!experiment) {
    return (
      <div className="exp-message exp-message-info">
        <Info size={21} />
        <div>
          <strong>Chưa có thông tin thí nghiệm</strong>
          <span>Hãy chọn một mô hình và tập dữ liệu đã huấn luyện để xem cấu hình.</span>
        </div>
      </div>
    );
  }

  const hp = experiment.hyperparameters;

  if (!hp) {
    return (
      <div className="exp-view-stack">
        <section className="exp-section">
          <SectionHeading
            icon={Settings}
            title="Cấu hình huấn luyện cơ bản (Legacy)"
            description="Thông số huấn luyện tiêu chuẩn của phiên bản V1 / V2."
          />
          <div className="exp-message exp-message-info" style={{ marginTop: 0 }}>
            <Info size={20} />
            <div>
              <strong>Thông báo hệ thống</strong>
              <span>Các phiên bản V1 và V2 sử dụng cấu hình huấn luyện tiêu chuẩn mặc định, không lưu trữ file tham số chi tiết.</span>
            </div>
          </div>
          <dl className="exp-training-summary" style={{ marginTop: 15 }}>
            <div>
              <dt>Trình tối ưu (Optimizer)</dt>
              <dd>{experiment.model.id === "yolo" ? "SGD (mặc định)" : "Adam (mặc định)"}</dd>
            </div>
            <div>
              <dt>Kích thước Batch</dt>
              <dd>32</dd>
            </div>
            <div>
              <dt>Hệ số học tập (Learning Rate)</dt>
              <dd>{experiment.model.id === "yolo" ? "0.01" : "0.001"}</dd>
            </div>
            <div>
              <dt>Kích thước ảnh đầu vào</dt>
              <dd>224 × 224 px</dd>
            </div>
            <div>
              <dt>Hàm mất mát (Loss Function)</dt>
              <dd>CrossEntropyLoss</dd>
            </div>
            <div>
              <dt>Chiến lược giảm LR</dt>
              <dd>Cosine Annealing / StepLR</dd>
            </div>
          </dl>
        </section>
      </div>
    );
  }

  const isYolo = experiment.model.id === "yolo";

  // Standard params
  const optimizer = hp.optimizer || "Chưa rõ";
  const epochs = hp.max_epochs || hp.epochs || "Chưa rõ";
  const batchSize = hp.batch_size || hp.batch || "Chưa rõ";
  const patience = hp.patience || "Không cài đặt";
  const seed = hp.seed !== undefined ? hp.seed : "Chưa rõ";
  
  // Learning rates
  let lrDisplay = "";
  if (hp.lr_head !== undefined || hp.lr_backbone !== undefined) {
    lrDisplay = `Head: ${hp.lr_head || "—"} | Backbone: ${hp.lr_backbone || "—"}`;
  } else if (hp.lr0 !== undefined) {
    lrDisplay = `LR Khởi đầu: ${hp.lr0} ${hp.lrf ? `(lrf: ${hp.lrf})` : ""}`;
  } else {
    lrDisplay = "Mặc định";
  }

  // Advanced Loss: Class-Balanced Loss
  const cbBeta = hp.cb_beta;
  const cbGamma = hp.cb_gamma;
  const isCbEnabled = cbBeta !== undefined || hp.loss === "ClassBalancedFocalLoss";

  // PML
  const pml = hp.pair_margin_loss;
  const isPmlEnabled = pml && pml.enabled;

  return (
    <div className="exp-view-stack">
      <div className="exp-kpi-grid">
        <div className="exp-kpi">
          <span>Trình tối ưu</span>
          <strong>{optimizer}</strong>
          <small>Phương thức tối ưu trọng số</small>
        </div>
        <div className="exp-kpi">
          <span>Batch Size</span>
          <strong>{batchSize}</strong>
          <small>Số ảnh / batch huấn luyện</small>
        </div>
        <div className="exp-kpi">
          <span>Epoch tối đa</span>
          <strong>{epochs}</strong>
          <small>Số epoch huấn luyện giới hạn</small>
        </div>
        <div className="exp-kpi">
          <span>Random Seed</span>
          <strong>{seed}</strong>
          <small>Seed tạo số ngẫu nhiên</small>
        </div>
      </div>

      <div className="exp-config-row-grid">
        {/* Basic Configuration Section */}
        <section className="exp-section">
          <SectionHeading
            icon={Settings}
            title="Thông số huấn luyện cơ bản"
            description="Các thông số điều phối và tối ưu hóa trong quá trình train."
          />
          <dl className="exp-training-summary">
            <div>
              <dt>Tốc độ học (Learning Rate)</dt>
              <dd>{lrDisplay}</dd>
            </div>
            <div>
              <dt>Dừng sớm (Patience)</dt>
              <dd>{patience} epoch</dd>
            </div>
            <div>
              <dt>Trọng số suy hao (Weight Decay)</dt>
              <dd>{hp.weight_decay !== undefined ? hp.weight_decay : "Mặc định"}</dd>
            </div>
            <div>
              <dt>Momentum</dt>
              <dd>{hp.momentum !== undefined ? hp.momentum : "Mặc định"}</dd>
            </div>
            <div>
              <dt>Dropout Rate</dt>
              <dd>{hp.dropout !== undefined ? `${(hp.dropout * 100).toFixed(0)}%` : "0%"}</dd>
            </div>
            <div>
              <dt>Kích thước ảnh</dt>
              <dd>{hp.imgsz !== undefined ? `${hp.imgsz} × ${hp.imgsz} px` : "224 × 224 px"}</dd>
            </div>
            {isYolo && (
              <>
                <div>
                  <dt>Tự động AMP</dt>
                  <dd>{hp.amp ? "Bật" : "Tắt"}</dd>
                </div>
                <div>
                  <dt>Deterministic</dt>
                  <dd>{hp.deterministic ? "Bật" : "Tắt"}</dd>
                </div>
              </>
            )}
          </dl>
        </section>

        {/* Custom Loss / Balance Section */}
        <section className="exp-section">
          <SectionHeading
            icon={ShieldCheck}
            title="Cấu hình Loss & Cân bằng lớp"
            description="Các thuật toán giải quyết bài toán mất cân bằng dữ liệu."
          />
          <dl className="exp-training-summary">
            <div>
              <dt>Class-Balanced Loss</dt>
              <dd>{isCbEnabled ? "Kích hoạt" : "Không áp dụng"}</dd>
            </div>
            {isCbEnabled && (
              <>
                <div>
                  <dt>CB Beta (Tần suất lớp)</dt>
                  <dd>{cbBeta || "0.999"}</dd>
                </div>
                <div>
                  <dt>Focal Loss Gamma</dt>
                  <dd>{cbGamma || "2.0"}</dd>
                </div>
              </>
            )}
            <div>
              <dt>Pairwise Margin Loss</dt>
              <dd>{isPmlEnabled ? "Kích hoạt" : "Không áp dụng"}</dd>
            </div>
            {isPmlEnabled && (
              <>
                <div>
                  <dt>Hệ số PML (Lambda)</dt>
                  <dd>{pml.lambda}</dd>
                </div>
                <div>
                  <dt>Biên độ PML (Margin)</dt>
                  <dd>{pml.margin}</dd>
                </div>
                <div style={{ gridColumn: "span 2" }}>
                  <dt>Cặp lớp áp dụng PML</dt>
                  <dd style={{ marginTop: "4px" }}>
                    {pml.classes && pml.classes.map((pair, idx) => (
                      <span key={idx} style={{ 
                        display: "inline-block", 
                        padding: "2px 8px", 
                        background: "#edf4fe", 
                        color: "#2563eb", 
                        borderRadius: "4px", 
                        fontSize: "12px",
                        marginRight: "6px",
                        border: "1px solid #bfdbfe",
                        fontWeight: "bold"
                      }}>
                        {pair.join(" ↔ ")}
                      </span>
                    ))}
                  </dd>
                </div>
              </>
            )}
            <div>
              <dt>Label Smoothing</dt>
              <dd>{hp.label_smoothing !== undefined ? hp.label_smoothing : (hp.cb_focal_label_smoothing !== undefined ? hp.cb_focal_label_smoothing : "0.0")}</dd>
            </div>
            <div>
              <dt>Mixup Alpha</dt>
              <dd>{hp.mixup_alpha !== undefined ? hp.mixup_alpha : (hp.mixup !== undefined ? hp.mixup : "Không áp dụng")}</dd>
            </div>
            {!isYolo && (
              <div>
                <dt>Cutmix Alpha</dt>
                <dd>{hp.cutmix_alpha !== undefined ? hp.cutmix_alpha : "Không áp dụng"}</dd>
              </div>
            )}
          </dl>
        </section>
      </div>

      {/* Class Counts details for Class Balanced Loss */}
      {hp.class_balanced_counts && (
        <details className="exp-details">
          <summary>Chi tiết số lượng mẫu tính Class-Balanced Weights</summary>
          <div className="exp-table-scroll">
            <table className="exp-table">
              <thead>
                <tr>
                  <th style={{ textAlign: "left" }}>Lớp xe</th>
                  <th>Số lượng ảnh làm trọng số (Effective Counts)</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(hp.class_balanced_counts).map(([cls, count]) => (
                  <tr key={cls}>
                    <th style={{ fontWeight: "600", textAlign: "left" }}>{cls}</th>
                    <td>{Number(count).toLocaleString("vi-VN")} ảnh</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}

      {/* Full raw configurations dropdown */}
      <details className="exp-details">
        <summary>Xem toàn bộ file cấu hình JSON/YAML thô</summary>
        <div style={{ 
          marginTop: "12px", 
          padding: "15px", 
          background: "#1e293b", 
          color: "#f8fafc", 
          borderRadius: "6px", 
          fontSize: "12px", 
          fontFamily: "monospace",
          whiteSpace: "pre-wrap",
          maxHeight: "350px",
          overflowY: "auto"
        }}>
          {JSON.stringify(hp, null, 2)}
        </div>
      </details>
    </div>
  );
}

function Dashboard() {
  const [catalog, setCatalog] = useState(null);
  const [modelId, setModelId] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [versionId, setVersionId] = useState("");
  const [activeView, setActiveView] = useState("overview");
  const [experiment, setExperiment] = useState(null);
  const [datasetProfile, setDatasetProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchExperimentCatalog()
      .then((data) => {
        if (cancelled) return;
        setCatalog(data);
        const firstReady = data.combinations.find((item) => item.available) || data.combinations[0];
        setModelId(firstReady?.model || "");
        setDatasetId(firstReady?.dataset || "");
        setVersionId(firstReady?.version || "");
      })
      .catch((requestError) => { if (!cancelled) setError(requestError.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [reloadKey]);

  const selectedCombination = catalog?.combinations.find((item) => item.model === modelId && item.dataset === datasetId && item.version === versionId);

  useEffect(() => {
    let cancelled = false;
    if (!catalog || !datasetId || !versionId) return undefined;
    setLoading(true);
    setError("");
    setExperiment(null);
    Promise.all([
      fetchDatasetProfile(datasetId, versionId),
      selectedCombination?.available ? fetchExperiment(selectedCombination.id) : Promise.resolve(null),
    ])
      .then(([profile, result]) => {
        if (cancelled) return;
        setDatasetProfile(profile);
        setExperiment(result);
      })
      .catch((requestError) => { if (!cancelled) setError(requestError.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [catalog, datasetId, versionId, selectedCombination?.id, selectedCombination?.available]);

  if (loading && !catalog) return <div className="experiment-dashboard"><LoadingState /></div>;
  if (error && !catalog) return <div className="experiment-dashboard"><ErrorState message={error} onRetry={() => setReloadKey((value) => value + 1)} /></div>;
  if (!catalog) return null;

  return (
    <section className="experiment-dashboard">
      <header className="exp-context-bar">
        <div className="exp-context-title"><span>Ngữ cảnh thí nghiệm</span><strong>Model × Dataset × Version</strong></div>
        <label>Model<select value={modelId} onChange={(event) => setModelId(event.target.value)}>{catalog.models.map((item) => <option value={item.id} key={item.id}>{item.label}</option>)}</select></label>
        <label>Dataset<select value={datasetId} onChange={(event) => setDatasetId(event.target.value)}>{catalog.datasets.map((item) => <option value={item.id} key={item.id}>{item.label}</option>)}</select></label>
        <label>Version<select value={versionId} onChange={(event) => setVersionId(event.target.value)}>{catalog.versions.map((item) => <option value={item.id} key={item.id}>{item.label}</option>)}</select></label>
        <div className={`exp-status ${selectedCombination?.available ? "ready" : "missing"}`}>
          {selectedCombination?.available ? <CheckCircle2 size={17} /> : <Info size={17} />}
          <span>{selectedCombination?.available ? "Đã có kết quả" : "Chưa train"}</span>
        </div>
      </header>

      <div className="exp-context-description">
        <Database size={16} /><span>{getItem(catalog.datasets, datasetId)?.description}</span>
        <span className="exp-dot" />
        <span>{getItem(catalog.versions, versionId)?.description}</span>
      </div>

      <nav className="exp-tabs" aria-label="Các màn hình dashboard">
        <button type="button" className={activeView === "overview" ? "active" : ""} onClick={() => setActiveView("overview")}><BarChart3 size={17} />Tổng quan</button>
        <button type="button" className={activeView === "analysis" ? "active" : ""} onClick={() => setActiveView("analysis")}><Grid3X3 size={17} />Phân tích</button>
        <button type="button" className={activeView === "config" ? "active" : ""} onClick={() => setActiveView("config")}><Settings size={17} />Cấu hình Train</button>
        <button type="button" className={activeView === "compare" ? "active" : ""} onClick={() => setActiveView("compare")}><GitCompareArrows size={17} />So sánh</button>
      </nav>

      {error ? <ErrorState message={error} /> : null}
      {loading && catalog ? <LoadingState /> : null}
      {!loading && !error && activeView === "overview" ? <OverviewView experiment={experiment} datasetProfile={datasetProfile} /> : null}
      {!loading && !error && activeView === "analysis" ? <AnalysisView experiment={experiment} /> : null}
      {!loading && !error && activeView === "config" ? <ConfigView experiment={experiment} /> : null}
      {!loading && activeView === "compare" ? <CompareView catalog={catalog} /> : null}
    </section>
  );
}

export default Dashboard;
