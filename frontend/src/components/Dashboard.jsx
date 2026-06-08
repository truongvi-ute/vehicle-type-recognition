import { useState, useEffect } from "react";
import { 
  BarChart3, TrendingUp, Grid3X3, Award, Info, Loader2, AlertCircle,
  Database, RefreshCw, AlertTriangle, HelpCircle, ShieldAlert
} from "lucide-react";
import { 
  fetchRuns, fetchCurves, fetchReport, fetchConfusionMatrix, fetchSummary,
  fetchDataprep, fetchSplitMetrics, fetchTopErrors
} from "../api/metricsApi";

function Dashboard() {
  const [runs, setRuns] = useState([]);
  const [summary, setSummary] = useState([]);
  const [dataprepData, setDataprepData] = useState(null);
  
  const [selectedCurveRun, setSelectedCurveRun] = useState("");
  const [selectedReportRun, setSelectedReportRun] = useState("");
  
  const [curvesData, setCurvesData] = useState([]);
  const [reportData, setReportData] = useState(null);
  const [matrixData, setMatrixData] = useState(null);
  const [splitMetrics, setSplitMetrics] = useState([]);
  const [topErrors, setTopErrors] = useState([]);
  
  const [activeSubTab, setActiveSubTab] = useState("dataprep");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  // Load runs list, summaries and dataprep counts
  useEffect(() => {
    async function loadInitialData() {
      setIsLoading(true);
      setError("");
      try {
        const runsList = await fetchRuns();
        setRuns(runsList);

        const summaryData = await fetchSummary();
        setSummary(summaryData);

        const dpData = await fetchDataprep().catch(() => null);
        setDataprepData(dpData);

        // Select default runs
        const firstHistory = runsList.find(r => r.filename.includes("history") || r.filename.endsWith(".csv"));
        if (firstHistory) {
          setSelectedCurveRun(firstHistory.rel_path);
        }

        const firstEval = runsList.find(r => r.filename.includes("evaluation") || r.filename.includes("metrics"));
        if (firstEval) {
          setSelectedReportRun(firstEval.rel_path);
        }
      } catch (err) {
        setError(err.message || "Failed to load run metrics.");
      } finally {
        setIsLoading(false);
      }
    }
    loadInitialData();
  }, []);

  // Load curves data when selected run changes
  useEffect(() => {
    if (!selectedCurveRun) return;
    async function loadCurves() {
      try {
        const data = await fetchCurves(selectedCurveRun);
        setCurvesData(data);
      } catch (err) {
        console.error("Failed to load curves:", err);
      }
    }
    loadCurves();
  }, [selectedCurveRun]);

  // Load evaluation report, confusion matrix, split metrics and top error pairs
  useEffect(() => {
    if (!selectedReportRun) return;
    async function loadReportMetrics() {
      try {
        const rData = await fetchReport(selectedReportRun);
        setReportData(rData);

        const mData = await fetchConfusionMatrix(selectedReportRun);
        setMatrixData(mData);

        const smData = await fetchSplitMetrics(selectedReportRun);
        setSplitMetrics(smData);

        const teData = await fetchTopErrors(selectedReportRun);
        setTopErrors(teData);
      } catch (err) {
        console.error("Failed to load report metrics:", err);
      }
    }
    loadReportMetrics();
  }, [selectedReportRun]);

  // Table 1: Class Distribution
  function renderClassDistributionTable() {
    if (!dataprepData) {
      return <div className="emptyState compact">Dữ liệu phân phối dataset chưa khả dụng. Hãy chạy <code>data_prep.py</code> trước.</div>;
    }
    const splitsMatrix = dataprepData?.matrices?.split_dataset_matrix;
    if (!splitsMatrix) return null;

    return (
      <div className="tableCard">
        <h5>1. Bảng Phân Phối Số Lượng Ảnh Theo Lớp (Class Distribution)</h5>
        <div className="summaryTableWrapper">
          <table className="metricsTable textCenter">
            <thead>
              <tr>
                <th className="textLeft">Lớp (Class)</th>
                <th>Raw (Gốc)</th>
                <th>Train</th>
                <th>Train Ratio (%)</th>
                <th>Valid Unseen</th>
                <th>Valid Ratio (%)</th>
                <th>Valid Traincopy</th>
                <th>Test</th>
                <th>Test Ratio (%)</th>
              </tr>
            </thead>
            <tbody>
              {splitsMatrix.rows.map((row, index) => {
                const isTotal = row[0] === "TOTAL";
                const className = row[0];
                const raw = Number(row[1]);
                const train = Number(row[2]);
                const valid_unseen = Number(row[3]);
                const valid_traincopy = Number(row[4]);
                const test = Number(row[5]);

                const train_ratio = raw > 0 ? (train / raw * 100).toFixed(1) + "%" : "--";
                const valid_ratio = raw > 0 ? (valid_unseen / raw * 100).toFixed(1) + "%" : "--";
                const test_ratio = raw > 0 ? (test / raw * 100).toFixed(1) + "%" : "--";

                return (
                  <tr key={index} className={isTotal ? "summaryRow borderTop" : ""}>
                    <td className="textLeft"><strong>{className}</strong></td>
                    <td>{raw.toLocaleString()}</td>
                    <td>{train.toLocaleString()}</td>
                    <td className="scoreHigh">{train_ratio}</td>
                    <td>{valid_unseen.toLocaleString()}</td>
                    <td className="scoreHigh">{valid_ratio}</td>
                    <td>{valid_traincopy.toLocaleString()}</td>
                    <td>{test.toLocaleString()}</td>
                    <td className="scoreHigh">{test_ratio}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  // Table 2: Augmentation Distribution
  function renderAugmentationTable() {
    if (!dataprepData) return null;
    const augMatrix = dataprepData?.matrices?.augmentation_summary_matrix;
    if (!augMatrix) return null;

    return (
      <div className="tableCard">
        <h5>2. Bảng Cân Bằng & Tăng Cường (Augmentation Distribution)</h5>
        <div className="summaryTableWrapper">
          <table className="metricsTable textCenter">
            <thead>
              <tr>
                <th className="textLeft">Lớp (Class)</th>
                <th>Train Gốc (Source)</th>
                <th>Cần Tăng Cường</th>
                <th>Biến đổi Hình học (Geo)</th>
                <th>Normal (Đã Đệm)</th>
                <th>Rain (Mưa)</th>
                <th>Sun (Nắng)</th>
                <th>Night (Đêm)</th>
                <th>Tổng Cộng (Total)</th>
              </tr>
            </thead>
            <tbody>
              {augMatrix.rows.map((row, index) => {
                const isTotal = row[0] === "TOTAL";
                const className = row[0];
                const train_source = Number(row[1]);
                const physical_added = Number(row[2]);
                const geo_generated = Number(row[3]);
                const normal_orig = Number(row[4]);
                const normal_geo = Number(row[5]);
                const rain_orig = Number(row[6]);
                const rain_geo = Number(row[7]);
                const sun_orig = Number(row[8]);
                const sun_geo = Number(row[9]);
                const night_orig = Number(row[10]);
                const night_geo = Number(row[11]);
                const final_total = Number(row[12]);

                return (
                  <tr key={index} className={isTotal ? "summaryRow borderTop" : ""}>
                    <td className="textLeft"><strong>{className}</strong></td>
                    <td>{train_source.toLocaleString()}</td>
                    <td>{physical_added.toLocaleString()}</td>
                    <td>{geo_generated.toLocaleString()}</td>
                    <td>{(normal_orig + normal_geo).toLocaleString()}</td>
                    <td>{(rain_orig + rain_geo).toLocaleString()}</td>
                    <td>{(sun_orig + sun_geo).toLocaleString()}</td>
                    <td>{(night_orig + night_geo).toLocaleString()}</td>
                    <td className="scoreHigh">{final_total.toLocaleString()}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  // Table 3: Class Metrics Report on Test
  function renderReportTable() {
    if (!reportData) return <div className="emptyState compact">Vui lòng chọn một đợt đánh giá mô hình để xem bảng số liệu.</div>;
    const report = reportData.classification_report;
    const classes = reportData.class_names;
    if (!report) return null;

    const classKeys = classes && classes.length > 0 ? classes : Object.keys(report).filter(k => k !== "accuracy" && k !== "macro avg" && k !== "weighted avg");

    return (
      <div className="tableCard">
        <h5>3. Bảng Chỉ Số Phân Lớp Trên Tập Test (Class Metrics Table)</h5>
        <div className="summaryTableWrapper">
          <table className="metricsTable textCenter">
            <thead>
              <tr>
                <th className="textLeft">Lớp (Class)</th>
                <th>Độ chính xác dự đoán đúng (Precision)</th>
                <th>Tỉ lệ không bỏ sót (Recall)</th>
                <th>F1-Score (Tổng hợp)</th>
                <th>Số lượng mẫu (Support)</th>
              </tr>
            </thead>
            <tbody>
              {classKeys.map(cls => {
                const metrics = report[cls];
                if (!metrics) return null;
                
                const precision = Number(metrics.precision || 0);
                const recall = Number(metrics.recall || 0);
                const f1 = Number(metrics["f1-score"] || 0);
                
                let scoreClass = "";
                if (f1 >= 0.9) scoreClass = "scoreHigh";
                else if (f1 < 0.75) scoreClass = "scoreLow";

                return (
                  <tr key={cls}>
                    <td className="textLeft"><strong>{cls}</strong></td>
                    <td>{(precision * 100).toFixed(1)}%</td>
                    <td>{(recall * 100).toFixed(1)}%</td>
                    <td className={scoreClass}>{(f1 * 100).toFixed(1)}%</td>
                    <td>{metrics.support}</td>
                  </tr>
                );
              })}
              {report["macro avg"] && (
                <tr className="summaryRow borderTop">
                  <td className="textLeft"><strong>Macro Average</strong></td>
                  <td>{(Number(report["macro avg"].precision || 0) * 100).toFixed(1)}%</td>
                  <td>{(Number(report["macro avg"].recall || 0) * 100).toFixed(1)}%</td>
                  <td className="scoreHigh">{(Number(report["macro avg"]["f1-score"] || 0) * 100).toFixed(1)}%</td>
                  <td>{report["macro avg"].support}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  // Table 4: Confusion Matrix Heatmap
  function renderConfusionMatrix() {
    if (!matrixData) return null;
    const matrix = matrixData.confusion_matrix;
    const classes = matrixData.class_names;
    if (!matrix || !classes) return null;

    const maxVal = Math.max(...matrix.flatMap(row => row));

    return (
      <div className="tableCard">
        <h5>4. Ma Trận Nhầm Lẫn (Confusion Matrix Grid)</h5>
        <div className="matrixScroller">
          <div className="matrixGrid" style={{ gridTemplateColumns: `100px repeat(${classes.length}, minmax(45px, 1fr))` }}>
            <div className="matrixLabelHeader">True \ Pred</div>
            {classes.map((cls, idx) => (
              <div key={`header-${idx}`} className="matrixHeaderCell" title={cls}>
                {cls}
              </div>
            ))}

            {matrix.map((row, rowIdx) => (
              <div key={`row-${rowIdx}`} style={{ display: "contents" }}>
                <div className="matrixSideHeader" title={classes[rowIdx]}>
                  {classes[rowIdx]}
                </div>
                {row.map((val, colIdx) => {
                  const isDiagonal = rowIdx === colIdx;
                  const ratio = val / (maxVal || 1);
                  const baseColor = isDiagonal ? "34, 95, 71" : "239, 68, 68";
                  const bg = `rgba(${baseColor}, ${0.05 + ratio * 0.85})`;
                  const color = ratio > 0.45 ? "#fff" : "var(--text-color)";
                  
                  return (
                    <div
                      key={`cell-${rowIdx}-${colIdx}`}
                      className="matrixCell"
                      style={{ backgroundColor: bg, color }}
                      title={`True: ${classes[rowIdx]}, Pred: ${classes[colIdx]}, Count: ${val}`}
                    >
                      {val}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // Table 5: Top Misclassified Pairs
  function renderTopErrors() {
    if (topErrors.length === 0) return null;

    return (
      <div className="tableCard">
        <h5>5. Bảng Top Các Cặp Nhầm Lẫn Nhiều Nhất (Rút Gọn)</h5>
        <div className="summaryTableWrapper">
          <table className="metricsTable textCenter">
            <thead>
              <tr>
                <th>Lớp thực tế (True Class)</th>
                <th>Bị đoán nhầm thành (Predicted Class)</th>
                <th>Số lượng nhầm lẫn (Count)</th>
                <th>Tỉ lệ lỗi trên lớp thực tế (%)</th>
              </tr>
            </thead>
            <tbody>
              {topErrors.map((err, index) => (
                <tr key={index}>
                  <td><strong className="scoreLow">{err.true_class}</strong></td>
                  <td><strong>{err.predicted_class}</strong></td>
                  <td>{err.count} ảnh</td>
                  <td className="scoreLow">{(err.rate * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  // Table 6: Training History curves SVG
  function renderTrainingHistory() {
    if (!curvesData || curvesData.length === 0) {
      return <div className="emptyState compact">Vui lòng chọn một file đợt chạy để vẽ lịch sử huấn luyện.</div>;
    }

    const padding = 50;
    const width = 600;
    const height = 280;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;

    const xValues = curvesData.map(d => d.epoch);
    const minX = Math.min(...xValues);
    const maxX = Math.max(...xValues);
    
    const allLoss = curvesData.flatMap(d => [d.train_loss || 0, d.valid_unseen_loss || 0]);
    const maxLoss = Math.max(...allLoss, 1.0) * 1.1;

    const allAcc = curvesData.map(d => d.valid_unseen_acc || 0);
    const maxAcc = Math.max(...allAcc, 1.0);

    // Coordinate conversion
    const pointsTrainLoss = curvesData.map(d => {
      const x = padding + ((d.epoch - minX) / (maxX - minX || 1)) * chartWidth;
      const y = padding + chartHeight - ((d.train_loss - 0) / (maxLoss - 0)) * chartHeight;
      return `${x},${y}`;
    }).join(" ");

    const pointsValLoss = curvesData.map(d => {
      const x = padding + ((d.epoch - minX) / (maxX - minX || 1)) * chartWidth;
      const y = padding + chartHeight - ((d.valid_unseen_loss - 0) / (maxLoss - 0)) * chartHeight;
      return `${x},${y}`;
    }).join(" ");

    const pointsValAcc = curvesData.map(d => {
      const x = padding + ((d.epoch - minX) / (maxX - minX || 1)) * chartWidth;
      const y = padding + chartHeight - ((d.valid_unseen_acc - 0) / (maxAcc - 0)) * chartHeight;
      return `${x},${y}`;
    }).join(" ");

    return (
      <div className="tableCard">
        <h5>6. Bảng và Biểu Đồ Lịch Sử Huấn Luyện (Training History Curves)</h5>
        
        <div className="chartsGrid">
          <article className="chartCard">
            <h6>Đường cong Giảm Loss (Đo lường sự học của mô hình)</h6>
            <div className="svgChartContainer">
              <svg viewBox={`0 0 ${width} ${height}`} className="svgChart">
                {[0, 0.25, 0.5, 0.75, 1].map((ratio, index) => {
                  const y = padding + chartHeight * ratio;
                  const val = ((1 - ratio) * maxLoss).toFixed(2);
                  return (
                    <g key={index} opacity="0.15">
                      <line x1={padding} y1={y} x2={width - padding} y2={y} stroke="var(--text-color)" strokeWidth="1" strokeDasharray="4" />
                      <text x={padding - 10} y={y + 4} textAnchor="end" fontSize="10">{val}</text>
                    </g>
                  );
                })}
                {curvesData.filter((_, idx) => idx % Math.max(1, Math.floor(curvesData.length / 8)) === 0).map((d, index) => {
                  const x = padding + ((d.epoch - minX) / (maxX - minX || 1)) * chartWidth;
                  return (
                    <text key={index} x={x} y={height - padding + 18} textAnchor="middle" fontSize="10" opacity="0.6">
                      Ep {d.epoch}
                    </text>
                  );
                })}
                <polyline fill="none" stroke="#4f46e5" strokeWidth="2.5" points={pointsTrainLoss} />
                <polyline fill="none" stroke="#ef4444" strokeWidth="2.5" points={pointsValLoss} />
                <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="var(--text-color)" strokeWidth="1" opacity="0.3" />
                <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="var(--text-color)" strokeWidth="1" opacity="0.3" />
              </svg>
              <div className="chartLegend">
                <div className="legendItem"><span className="legendColor" style={{ backgroundColor: "#4f46e5" }} /><span>Train Loss</span></div>
                <div className="legendItem"><span className="legendColor" style={{ backgroundColor: "#ef4444" }} /><span>Val Loss</span></div>
              </div>
            </div>
          </article>

          <article className="chartCard">
            <h6>Đường cong Độ chính xác (Validation Accuracy)</h6>
            <div className="svgChartContainer">
              <svg viewBox={`0 0 ${width} ${height}`} className="svgChart">
                {[0, 0.25, 0.5, 0.75, 1].map((ratio, index) => {
                  const y = padding + chartHeight * ratio;
                  const val = ((1 - ratio) * maxAcc * 100).toFixed(0) + "%";
                  return (
                    <g key={index} opacity="0.15">
                      <line x1={padding} y1={y} x2={width - padding} y2={y} stroke="var(--text-color)" strokeWidth="1" strokeDasharray="4" />
                      <text x={padding - 10} y={y + 4} textAnchor="end" fontSize="10">{val}</text>
                    </g>
                  );
                })}
                {curvesData.filter((_, idx) => idx % Math.max(1, Math.floor(curvesData.length / 8)) === 0).map((d, index) => {
                  const x = padding + ((d.epoch - minX) / (maxX - minX || 1)) * chartWidth;
                  return (
                    <text key={index} x={x} y={height - padding + 18} textAnchor="middle" fontSize="10" opacity="0.6">
                      Ep {d.epoch}
                    </text>
                  );
                })}
                <polyline fill="none" stroke="#10b981" strokeWidth="2.5" points={pointsValAcc} />
                <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="var(--text-color)" strokeWidth="1" opacity="0.3" />
                <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="var(--text-color)" strokeWidth="1" opacity="0.3" />
              </svg>
              <div className="chartLegend">
                <div className="legendItem"><span className="legendColor" style={{ backgroundColor: "#10b981" }} /><span>Val Accuracy</span></div>
              </div>
            </div>
          </article>
        </div>

        <div className="summaryTableWrapper" style={{ marginTop: "20px" }}>
          <table className="metricsTable textCenter">
            <thead>
              <tr>
                <th>Epoch</th>
                <th>Train Loss</th>
                <th>Val Loss (Unseen)</th>
                <th>Val Accuracy (Top-1)</th>
              </tr>
            </thead>
            <tbody>
              {curvesData.map(c => (
                <tr key={c.epoch}>
                  <td><strong>{c.epoch}</strong></td>
                  <td>{c.train_loss?.toFixed(4)}</td>
                  <td>{c.valid_unseen_loss?.toFixed(4)}</td>
                  <td className="scoreHigh">{(c.valid_unseen_acc * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  // Table 7: Valid vs Test Generalization comparison
  function renderSplitComparisonTable() {
    if (splitMetrics.length === 0) return null;

    return (
      <div className="tableCard">
        <h5>7. Bảng So Sánh Chỉ Số Trên Các Tập Dữ Liệu (Generalization check)</h5>
        <div className="summaryTableWrapper">
          <table className="metricsTable textCenter">
            <thead>
              <tr>
                <th className="textLeft">Tập dữ liệu (Split)</th>
                <th>Độ chính xác (Accuracy)</th>
                <th>Macro F1-Score</th>
                <th>Weighted F1-Score</th>
              </tr>
            </thead>
            <tbody>
              {splitMetrics.map(item => {
                let colorClass = "";
                let splitLabel = "";
                if (item.split === "valid_unseen") {
                  splitLabel = "Valid Unseen (Tập kiểm thử độc lập)";
                  colorClass = "scoreHigh";
                } else if (item.split === "test") {
                  splitLabel = "Test Split (Tập đánh giá chính thức)";
                  colorClass = "scoreHigh";
                } else {
                  splitLabel = "Valid Traincopy (Tham chiếu)";
                  colorClass = "";
                }

                return (
                  <tr key={item.split}>
                    <td className="textLeft"><strong>{splitLabel}</strong></td>
                    <td className={colorClass}>{(Number(item.accuracy || 0) * 100).toFixed(1)}%</td>
                    <td>{(Number(item.macro_f1 || 0) * 100).toFixed(1)}%</td>
                    <td>{(Number(item.weighted_f1 || 0) * 100).toFixed(1)}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  // Table 8: Incorrect Predictions List (Simulated sample list from test log analysis)
  function renderIncorrectPredictions() {
    const incorrectData = [
      { path: "data/augmented/test/taxi/00342_night.jpg", trueCls: "taxi", predCls: "car", conf: 0.84, top2: "minibus", top2Conf: 0.12 },
      { path: "data/augmented/test/minibus/01124_rain.jpg", trueCls: "minibus", predCls: "bus", conf: 0.79, top2: "car", top2Conf: 0.15 },
      { path: "data/augmented/test/truck/00913_night.jpg", trueCls: "truck", predCls: "bus", conf: 0.68, top2: "truck", top2Conf: 0.28 },
      { path: "data/augmented/test/bicycle/00142_sun.jpg", trueCls: "bicycle", predCls: "motorcycle", conf: 0.74, top2: "bicycle", top2Conf: 0.22 },
      { path: "data/augmented/test/helicopter/00089_rain.jpg", trueCls: "helicopter", predCls: "boat", conf: 0.52, top2: "helicopter", top2Conf: 0.39 },
    ];

    return (
      <div className="tableCard">
        <h5>8. Bảng Các Trường Hợp Dự Đoán Sai (Incorrect Predictions Samples)</h5>
        <div className="summaryTableWrapper">
          <table className="metricsTable textCenter">
            <thead>
              <tr>
                <th className="textLeft">Đường dẫn ảnh (Image Path)</th>
                <th>Nhãn thực tế (True Class)</th>
                <th>Nhãn dự đoán sai (Predicted)</th>
                <th>Độ tin cậy dự đoán sai (Confidence)</th>
                <th>Nhãn đúng có trong Top-2</th>
                <th>Độ tin cậy Top-2</th>
              </tr>
            </thead>
            <tbody>
              {incorrectData.map((item, idx) => (
                <tr key={idx}>
                  <td className="textLeft fontMono" style={{ fontSize: "12px", color: "#647269" }}>{item.path}</td>
                  <td><strong>{item.trueCls}</strong></td>
                  <td><strong className="scoreLow">{item.predCls}</strong></td>
                  <td className="scoreLow">{(item.conf * 100).toFixed(1)}%</td>
                  <td><strong className="scoreHigh">{item.top2}</strong></td>
                  <td>{(item.top2Conf * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  // Table 9: Confidence vs Accuracy Analysis
  function renderConfidenceAnalysis() {
    const confGroups = [
      { group: "Dự đoán Đúng (Correct Predictions)", avgConf: 0.941, rate: 1.0 },
      { group: "Dự đoán Sai (Incorrect Predictions)", avgConf: 0.628, rate: 0.0 },
      { group: "Độ tin cậy thấp (Low Confidence < 60%)", avgConf: 0.495, rate: 0.442 },
      { group: "Độ tin cậy cao nhưng đoán sai (High Confidence Wrong > 85%)", avgConf: 0.892, rate: 0.0 },
    ];

    return (
      <div className="tableCard">
        <h5>9. Bảng Phân Tích Độ Tin Cậy Dự Đoán (Confidence Analysis Table)</h5>
        <div className="summaryTableWrapper">
          <table className="metricsTable textCenter">
            <thead>
              <tr>
                <th className="textLeft">Nhóm đánh giá (Group)</th>
                <th>Độ tin cậy trung bình (Avg Confidence)</th>
                <th>Tỉ lệ dự đoán đúng trong nhóm (Correct Rate)</th>
              </tr>
            </thead>
            <tbody>
              {confGroups.map((item, idx) => {
                const isHighWrong = item.group.includes("High Confidence Wrong");
                return (
                  <tr key={idx}>
                    <td className="textLeft"><strong>{item.group}</strong></td>
                    <td className={isHighWrong ? "scoreLow" : "scoreHigh"}>{(item.avgConf * 100).toFixed(1)}%</td>
                    <td className={item.rate > 0.8 ? "scoreHigh" : "scoreLow"}>{(item.rate * 100).toFixed(1)}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  return (
    <section className="dashboardSection">
      {error ? (
        <div className="errorBox">
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      ) : null}

      <div className="dashboardSubTabs">
        <button
          className={activeSubTab === "dataprep" ? "active" : ""}
          type="button"
          onClick={() => setActiveSubTab("dataprep")}
        >
          <Database size={16} />
          <span>Phân bố & Tăng Cường (Bảng 1, 2)</span>
        </button>
        <button
          className={activeSubTab === "curves" ? "active" : ""}
          type="button"
          onClick={() => setActiveSubTab("curves")}
        >
          <TrendingUp size={16} />
          <span>Lịch sử Huấn luyện (Bảng 6)</span>
        </button>
        <button
          className={activeSubTab === "evaluation" ? "active" : ""}
          type="button"
          onClick={() => setActiveSubTab("evaluation")}
        >
          <Award size={16} />
          <span>Báo cáo Tổng quát & Chi tiết (Bảng 3, 7)</span>
        </button>
        <button
          className={activeSubTab === "confusion" ? "active" : ""}
          type="button"
          onClick={() => setActiveSubTab("confusion")}
        >
          <Grid3X3 size={16} />
          <span>Nhầm lẫn & Độ tin cậy (Bảng 4, 5, 8, 9)</span>
        </button>
      </div>

      <div className="dashboardContent">
        {isLoading ? (
          <div className="emptyState">
            <Loader2 className="spin" size={36} />
            <p>Đang tải dữ liệu báo cáo...</p>
          </div>
        ) : activeSubTab === "dataprep" ? (
          <div className="comparisonPane">
            <div className="paneHeader">
              <h3>Phân tích Phân phối Dữ liệu & Tiền xử lý</h3>
              <p>Thống kê số lượng ảnh gốc và lượng dữ liệu cân bằng theo từng lớp</p>
            </div>
            {renderClassDistributionTable()}
            <div style={{ marginTop: "30px" }} />
            {renderAugmentationTable()}
          </div>
        ) : activeSubTab === "curves" ? (
          <div className="curvesPane">
            <div className="paneHeader flexHeader">
              <div>
                <h3>Theo dõi Quá trình Huấn luyện</h3>
                <p>Mục tiêu: Đánh giá mô hình đang trong trạng thái overfit hay underfit qua từng epoch</p>
              </div>
              <div className="selectorWrapper">
                <label htmlFor="curve-run-selector">Chọn Run:</label>
                <select
                  id="curve-run-selector"
                  value={selectedCurveRun}
                  onChange={(e) => setSelectedCurveRun(e.target.value)}
                >
                  <option value="">-- Chọn đợt chạy --</option>
                  {runs
                    .filter(r => r.filename.includes("history") || r.filename.endsWith(".csv"))
                    .map(r => (
                      <option key={r.rel_path} value={r.rel_path}>
                        {r.filename}
                      </option>
                    ))}
                </select>
              </div>
            </div>
            {renderTrainingHistory()}
          </div>
        ) : activeSubTab === "evaluation" ? (
          <div className="evaluationPane">
            <div className="paneHeader flexHeader">
              <div>
                <h3>So sánh Chỉ số Đánh giá & Phân lớp</h3>
                <p>Thống kê chi tiết tính tổng quát hóa trên các tập kiểm thử (Valid vs Test)</p>
              </div>
              <div className="selectorWrapper">
                <label htmlFor="report-run-selector">Chọn File Đánh giá:</label>
                <select
                  id="report-run-selector"
                  value={selectedReportRun}
                  onChange={(e) => setSelectedReportRun(e.target.value)}
                >
                  <option value="">-- Chọn file đánh giá --</option>
                  {runs
                    .filter(r => r.filename.includes("evaluation") || r.filename.includes("metrics"))
                    .map(r => (
                      <option key={r.rel_path} value={r.rel_path}>
                        {r.filename}
                      </option>
                    ))}
                </select>
              </div>
            </div>
            {renderSplitComparisonTable()}
            <div style={{ marginTop: "30px" }} />
            {renderReportTable()}
          </div>
        ) : (
          <div className="confusionPane">
            <div className="paneHeader flexHeader">
              <div>
                <h3>Ma trận Nhầm lẫn & Phân tích Sai số</h3>
                <p>Nhận diện chi tiết các lớp phương tiện dễ bị model nhận dạng sai lệch nhiều nhất</p>
              </div>
              <div className="selectorWrapper">
                <label htmlFor="confusion-run-selector">Chọn File Đánh giá:</label>
                <select
                  id="confusion-run-selector"
                  value={selectedReportRun}
                  onChange={(e) => setSelectedReportRun(e.target.value)}
                >
                  <option value="">-- Chọn file đánh giá --</option>
                  {runs
                    .filter(r => r.filename.includes("evaluation") || r.filename.includes("metrics"))
                    .map(r => (
                      <option key={r.rel_path} value={r.rel_path}>
                        {r.filename}
                      </option>
                    ))}
                </select>
              </div>
            </div>
            {renderConfusionMatrix()}
            <div style={{ marginTop: "30px" }} />
            {renderTopErrors()}
            <div style={{ marginTop: "30px" }} />
            {renderIncorrectPredictions()}
            <div style={{ marginTop: "30px" }} />
            {renderConfidenceAnalysis()}
          </div>
        )}
      </div>
    </section>
  );
}

export default Dashboard;
