const defaultDatasetPath = "data/diabetes_prediction_dataset.csv";

const demoCsv = `gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes
Female,42,0,0,never,24.1,5.4,112,0
Male,58,1,0,former,31.8,7.2,190,1
Female,36,0,0,No Info,22.7,5.1,98,0
Male,61,1,1,current,34.5,8.1,224,1
Female,49,0,0,former,29.2,6.4,166,1
Male,29,0,0,never,23.8,4.9,96,0
Female,67,1,1,not current,33.4,7.8,210,1
Male,45,0,0,ever,28.1,5.8,134,0
Female,53,1,0,former,32.2,6.9,184,1
Male,39,0,0,never,25.6,5.3,110,0
Female,31,0,0,never,21.9,4.8,92,0
Male,72,1,1,former,30.7,8.4,236,1
Female,64,1,0,current,35.1,7.6,206,1
Male,51,0,0,not current,27.2,5.9,142,0
Female,47,0,0,ever,26.8,5.7,128,0
Male,55,1,0,former,33.5,7.1,188,1
Female,26,0,0,never,20.3,4.7,88,0
Male,43,0,0,current,29.4,6.1,156,1
Female,59,1,0,former,31.6,7.0,179,1
Male,34,0,0,never,24.7,5.0,101,0
Female,62,1,1,not current,36.4,8.2,230,1
Male,41,0,0,ever,27.5,5.6,126,0
Female,57,1,0,former,30.8,6.8,174,1
Male,38,0,0,No Info,26.1,5.2,108,0
Female,44,0,0,never,23.4,5.1,105,0
Male,66,1,1,current,35.8,8.0,218,1
Female,52,1,0,former,29.7,6.6,170,1
Male,28,0,0,never,22.9,4.8,94,0
Female,48,0,0,ever,27.9,5.9,138,0
Male,63,1,1,former,34.2,7.7,208,1
Female,35,0,0,never,24.6,5.0,99,0
Male,54,1,0,not current,32.8,6.9,181,1
Female,69,1,1,current,37.2,8.5,242,1
Male,46,0,0,former,28.7,6.0,148,0
Female,40,0,0,No Info,25.1,5.4,116,0
Male,60,1,1,ever,33.1,7.4,194,1`;

const statsGrid = document.getElementById("stats-grid");
const summaryInsights = document.getElementById("summary-insights");
const classBreakdown = document.getElementById("class-breakdown");
const genderChart = document.getElementById("gender-chart");
const smokingChart = document.getElementById("smoking-chart");
const glucoseChart = document.getElementById("glucose-chart");
const modelMetrics = document.getElementById("model-metrics");
const confusionMatrix = document.getElementById("confusion-matrix");
const featureImportance = document.getElementById("feature-importance");
const dataTableBody = document.getElementById("data-table-body");
const predictionOutput = document.getElementById("prediction-output");
const predictionForm = document.getElementById("prediction-form");
const datasetUpload = document.getElementById("dataset-upload");
const useDemoButton = document.getElementById("use-demo");
const datasetSource = document.getElementById("dataset-source");
const datasetSize = document.getElementById("dataset-size");
const modelStatus = document.getElementById("model-status");

const featureNames = [
  "age",
  "hypertension",
  "heart_disease",
  "bmi",
  "HbA1c_level",
  "blood_glucose_level",
  "gender_male",
  "gender_other",
  "smoke_former",
  "smoke_current",
  "smoke_not_current",
  "smoke_ever",
  "smoke_no_info"
];

const appState = {
  rows: [],
  source: "",
  model: null
};

function splitCsvLine(line) {
  const values = [];
  let current = "";
  let quoted = false;

  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (char === '"') {
      quoted = !quoted;
      continue;
    }
    if (char === "," && !quoted) {
      values.push(current.trim());
      current = "";
      continue;
    }
    current += char;
  }

  values.push(current.trim());
  return values;
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  const headers = splitCsvLine(lines[0] || "");
  return lines.slice(1).map((line) => {
    const cells = splitCsvLine(line);
    return headers.reduce((entry, header, index) => {
      entry[header] = cells[index] ?? "";
      return entry;
    }, {});
  });
}

function normalizeSmoking(value) {
  const clean = String(value || "No Info").trim().toLowerCase();
  if (clean === "former") return "former";
  if (clean === "current") return "current";
  if (clean === "not current") return "not current";
  if (clean === "ever") return "ever";
  if (clean === "no info" || clean === "n/a" || clean === "nan") return "No Info";
  return "never";
}

function normalizeGender(value) {
  const clean = String(value || "").trim().toLowerCase();
  if (clean === "male") return "Male";
  if (clean === "female") return "Female";
  return "Other";
}

function normalizeRow(row) {
  const normalized = {
    gender: normalizeGender(row.gender),
    age: Number(row.age),
    hypertension: Number(row.hypertension),
    heart_disease: Number(row.heart_disease),
    smoking_history: normalizeSmoking(row.smoking_history),
    bmi: Number(row.bmi),
    HbA1c_level: Number(row.HbA1c_level),
    blood_glucose_level: Number(row.blood_glucose_level),
    diabetes: Number(row.diabetes)
  };

  if (
    Number.isNaN(normalized.age) ||
    Number.isNaN(normalized.hypertension) ||
    Number.isNaN(normalized.heart_disease) ||
    Number.isNaN(normalized.bmi) ||
    Number.isNaN(normalized.HbA1c_level) ||
    Number.isNaN(normalized.blood_glucose_level) ||
    Number.isNaN(normalized.diabetes)
  ) {
    return null;
  }

  return normalized;
}

function average(values) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatNumber(value, digits = 1) {
  return Number(value).toLocaleString("en-IN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits
  });
}

function countBy(rows, selector) {
  return rows.reduce((counts, row) => {
    const key = selector(row);
    counts[key] = (counts[key] || 0) + 1;
    return counts;
  }, {});
}

function getGlucoseBand(value) {
  if (value < 100) return "Normal";
  if (value < 140) return "Elevated";
  if (value < 200) return "High";
  return "Critical";
}

function getRiskTone(probability) {
  if (probability >= 0.7) return "high";
  if (probability >= 0.4) return "moderate";
  return "low";
}

function renderStats(rows) {
  const positive = rows.filter((row) => row.diabetes === 1);
  const cards = [
    ["Patients Loaded", rows.length.toLocaleString("en-IN"), "Records currently available for analysis"],
    ["Diabetes Prevalence", formatPercent(positive.length / Math.max(rows.length, 1)), `${positive.length.toLocaleString("en-IN")} positive cases in the dataset`],
    ["Average BMI", formatNumber(average(rows.map((row) => row.bmi))), "Useful for obesity-linked risk comparison"],
    ["Average Glucose", formatNumber(average(rows.map((row) => row.blood_glucose_level)), 0), "Mean blood glucose level across the cohort"],
    ["Hypertension Share", formatPercent(rows.filter((row) => row.hypertension === 1).length / Math.max(rows.length, 1)), "Patients already carrying a major comorbidity"],
    ["High HbA1c Cases", rows.filter((row) => row.HbA1c_level >= 6.5).length.toLocaleString("en-IN"), "Records with HbA1c at diabetic-range threshold"],
    ["Senior Population", formatPercent(rows.filter((row) => row.age >= 60).length / Math.max(rows.length, 1)), "Patients aged 60 or above"],
    ["Critical Glucose", rows.filter((row) => row.blood_glucose_level >= 200).length.toLocaleString("en-IN"), "Patients in the highest glucose band"]
  ];

  statsGrid.innerHTML = cards.map(([label, value, note]) => `
    <article class="stat-card">
      <p class="eyebrow">${label}</p>
      <strong>${value}</strong>
      <p>${note}</p>
    </article>
  `).join("");
}

function renderSummaryInsights(rows) {
  const positive = rows.filter((row) => row.diabetes === 1);
  const negative = rows.filter((row) => row.diabetes === 0);
  const cards = [
    {
      title: "Age difference",
      body: `Patients with diabetes are older on average (${formatNumber(average(positive.map((row) => row.age)))} years) compared with non-diabetic records (${formatNumber(average(negative.map((row) => row.age)))} years).`
    },
    {
      title: "BMI effect",
      body: `The diabetic group shows a higher mean BMI (${formatNumber(average(positive.map((row) => row.bmi)))}) than the non-diabetic group (${formatNumber(average(negative.map((row) => row.bmi)))}).`
    },
    {
      title: "HbA1c separation",
      body: `HbA1c stands out clearly in this cohort, moving from ${formatNumber(average(negative.map((row) => row.HbA1c_level)))} in non-diabetic records to ${formatNumber(average(positive.map((row) => row.HbA1c_level)))} in diabetic cases.`
    }
  ];

  summaryInsights.innerHTML = cards.map((card) => `
    <div class="insight-card">
      <strong>${card.title}</strong>
      <p>${card.body}</p>
    </div>
  `).join("");
}

function renderBarList(element, counts, total) {
  element.innerHTML = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([label, count]) => `
      <div class="chart-row">
        <div class="chart-head"><span>${label}</span><span>${count.toLocaleString("en-IN")}</span></div>
        <div class="progress-track"><div class="progress-fill" style="width:${(count / Math.max(total, 1)) * 100}%"></div></div>
        <div class="chart-meta">${formatPercent(count / Math.max(total, 1))}</div>
      </div>
    `).join("");
}

function renderClassBalance(rows) {
  const counts = countBy(rows, (row) => row.diabetes === 1 ? "Diabetic" : "Non-diabetic");
  renderBarList(classBreakdown, counts, rows.length);
}

function renderCharts(rows) {
  renderBarList(genderChart, countBy(rows, (row) => row.gender), rows.length);
  renderBarList(smokingChart, countBy(rows, (row) => row.smoking_history), rows.length);
  renderBarList(glucoseChart, countBy(rows, (row) => getGlucoseBand(row.blood_glucose_level)), rows.length);
}

function seededRandom(seed) {
  let value = seed;
  return () => {
    value += 0x6d2b79f5;
    let result = Math.imul(value ^ (value >>> 15), 1 | value);
    result ^= result + Math.imul(result ^ (result >>> 7), 61 | result);
    return ((result ^ (result >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffleRows(rows) {
  const random = seededRandom(42);
  const copy = [...rows];
  for (let index = copy.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    [copy[index], copy[swapIndex]] = [copy[swapIndex], copy[index]];
  }
  return copy;
}

function encodeRow(row) {
  return [
    row.age,
    row.hypertension,
    row.heart_disease,
    row.bmi,
    row.HbA1c_level,
    row.blood_glucose_level,
    row.gender === "Male" ? 1 : 0,
    row.gender === "Other" ? 1 : 0,
    row.smoking_history === "former" ? 1 : 0,
    row.smoking_history === "current" ? 1 : 0,
    row.smoking_history === "not current" ? 1 : 0,
    row.smoking_history === "ever" ? 1 : 0,
    row.smoking_history === "No Info" ? 1 : 0
  ];
}

function fitScaler(features) {
  const means = new Array(features[0].length).fill(0);
  const stds = new Array(features[0].length).fill(0);

  for (let column = 0; column < features[0].length; column += 1) {
    const values = features.map((row) => row[column]);
    means[column] = average(values);
    const variance = average(values.map((value) => (value - means[column]) ** 2));
    stds[column] = Math.sqrt(variance) || 1;
  }

  return { means, stds };
}

function transformFeatures(features, scaler) {
  return features.map((row) => row.map((value, column) => (value - scaler.means[column]) / scaler.stds[column]));
}

function dotProduct(left, right) {
  let total = 0;
  for (let index = 0; index < left.length; index += 1) {
    total += left[index] * right[index];
  }
  return total;
}

function sigmoid(value) {
  if (value < -35) return 0;
  if (value > 35) return 1;
  return 1 / (1 + Math.exp(-value));
}

function trainLogisticRegression(features, labels) {
  const weights = new Array(features[0].length).fill(0);
  let bias = 0;

  for (let epoch = 0; epoch < 260; epoch += 1) {
    const gradients = new Array(weights.length).fill(0);
    let biasGradient = 0;

    for (let rowIndex = 0; rowIndex < features.length; rowIndex += 1) {
      const prediction = sigmoid(dotProduct(features[rowIndex], weights) + bias);
      const error = prediction - labels[rowIndex];
      for (let column = 0; column < weights.length; column += 1) {
        gradients[column] += error * features[rowIndex][column];
      }
      biasGradient += error;
    }

    for (let column = 0; column < weights.length; column += 1) {
      weights[column] -= (0.18 * gradients[column]) / features.length;
    }

    bias -= (0.18 * biasGradient) / features.length;
  }

  return { weights, bias };
}

function predictProbability(featureRow, model) {
  const standardized = featureRow.map((value, index) => (value - model.scaler.means[index]) / model.scaler.stds[index]);
  return sigmoid(dotProduct(standardized, model.weights) + model.bias);
}

function evaluateModel(testX, testY, model) {
  let tp = 0;
  let tn = 0;
  let fp = 0;
  let fn = 0;

  for (let index = 0; index < testX.length; index += 1) {
    const prediction = predictProbability(testX[index], model) >= 0.5 ? 1 : 0;
    const actual = testY[index];
    if (prediction === 1 && actual === 1) tp += 1;
    if (prediction === 0 && actual === 0) tn += 1;
    if (prediction === 1 && actual === 0) fp += 1;
    if (prediction === 0 && actual === 1) fn += 1;
  }

  const accuracy = (tp + tn) / Math.max(testY.length, 1);
  const precision = tp / Math.max(tp + fp, 1);
  const recall = tp / Math.max(tp + fn, 1);
  const f1 = (2 * precision * recall) / Math.max(precision + recall, 1e-8);

  return { accuracy, precision, recall, f1, tp, tn, fp, fn };
}

function buildModel(rows) {
  const sample = rows.length > 12000 ? shuffleRows(rows).slice(0, 12000) : shuffleRows(rows);
  const features = sample.map((row) => encodeRow(row));
  const labels = sample.map((row) => row.diabetes);
  const splitIndex = Math.max(Math.floor(sample.length * 0.8), Math.min(10, sample.length - 1));
  const trainXRaw = features.slice(0, splitIndex);
  const testXRaw = features.slice(splitIndex);
  const trainY = labels.slice(0, splitIndex);
  const testY = labels.slice(splitIndex);
  const scaler = fitScaler(trainXRaw);
  const trainX = transformFeatures(trainXRaw, scaler);
  const baseModel = trainLogisticRegression(trainX, trainY);
  const model = { ...baseModel, scaler };
  return { ...model, metrics: evaluateModel(testXRaw, testY, model) };
}

function renderModel(model) {
  const metrics = [
    ["Accuracy", formatPercent(model.metrics.accuracy)],
    ["Precision", formatPercent(model.metrics.precision)],
    ["Recall", formatPercent(model.metrics.recall)],
    ["F1 Score", formatPercent(model.metrics.f1)]
  ];

  modelMetrics.innerHTML = metrics.map(([label, value]) => `
    <div class="metric-box">
      <span>${label}</span>
      <strong>${value}</strong>
    </div>
  `).join("");

  const matrix = [
    ["True Positive", model.metrics.tp],
    ["False Positive", model.metrics.fp],
    ["False Negative", model.metrics.fn],
    ["True Negative", model.metrics.tn]
  ];

  confusionMatrix.innerHTML = matrix.map(([label, value]) => `
    <div class="matrix-cell">
      <span>${label}</span>
      <strong>${value.toLocaleString("en-IN")}</strong>
    </div>
  `).join("");

  const ranked = featureNames.map((name, index) => ({ name, weight: Math.abs(model.weights[index]) }))
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 6);
  const maxWeight = Math.max(...ranked.map((item) => item.weight), 1);

  featureImportance.innerHTML = ranked.map((item) => `
    <div class="chart-row">
      <div class="chart-head"><span>${item.name.replaceAll("_", " ")}</span><span>${item.weight.toFixed(2)}</span></div>
      <div class="progress-track"><div class="progress-fill" style="width:${(item.weight / maxWeight) * 100}%"></div></div>
    </div>
  `).join("");
}

function renderTable(rows) {
  dataTableBody.innerHTML = rows.slice(0, 12).map((row) => `
    <tr>
      <td>${row.gender}</td>
      <td>${row.age}</td>
      <td>${row.hypertension}</td>
      <td>${row.heart_disease}</td>
      <td>${row.smoking_history}</td>
      <td>${row.bmi}</td>
      <td>${row.HbA1c_level}</td>
      <td>${row.blood_glucose_level}</td>
      <td>${row.diabetes}</td>
    </tr>
  `).join("");
}

function renderPrediction(probability) {
  const tone = getRiskTone(probability);
  const message = tone === "high"
    ? "The profile falls in a high-risk zone and would need medical screening quickly."
    : tone === "moderate"
      ? "The profile shows moderate risk and should be reviewed with additional tests."
      : "The profile appears low risk in this model, though clinical confirmation is still needed.";

  predictionOutput.innerHTML = `
    <div class="prediction-card">
      <span class="eyebrow">Estimated diabetes probability</span>
      <strong>${formatPercent(probability)}</strong>
      <span class="risk-tone ${tone}">${tone.toUpperCase()} RISK</span>
      <p>${message}</p>
    </div>
  `;
}

function renderDashboard() {
  renderStats(appState.rows);
  renderSummaryInsights(appState.rows);
  renderClassBalance(appState.rows);
  renderCharts(appState.rows);
  renderTable(appState.rows);
  renderModel(appState.model);
  datasetSource.textContent = appState.source;
  datasetSize.textContent = appState.rows.length.toLocaleString("en-IN");
  modelStatus.textContent = "Model trained";
}

function loadRowsFromText(text, sourceLabel) {
  const rows = parseCsv(text).map((row) => normalizeRow(row)).filter(Boolean);
  if (!rows.length) {
    datasetSource.textContent = "Dataset could not be parsed";
    modelStatus.textContent = "No usable rows found";
    return;
  }

  appState.rows = rows;
  appState.source = sourceLabel;
  modelStatus.textContent = "Training model...";

  window.requestAnimationFrame(() => {
    appState.model = buildModel(rows);
    renderDashboard();
    renderPrediction(predictProbability(encodeRow(rows[0]), appState.model));
  });
}

async function loadInitialDataset() {
  try {
    const response = await fetch(defaultDatasetPath, { cache: "no-store" });
    if (!response.ok) throw new Error("Local dataset not found");
    loadRowsFromText(await response.text(), "Local Kaggle CSV from /data");
  } catch (error) {
    loadRowsFromText(demoCsv, "Built-in demo dataset");
  }
}

datasetUpload?.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  loadRowsFromText(await file.text(), `Uploaded file: ${file.name}`);
});

useDemoButton?.addEventListener("click", () => {
  loadRowsFromText(demoCsv, "Built-in demo dataset");
});

predictionForm?.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!appState.model) return;

  const formData = new FormData(predictionForm);
  const record = normalizeRow({
    gender: formData.get("gender"),
    age: formData.get("age"),
    hypertension: formData.get("hypertension"),
    heart_disease: formData.get("heart_disease"),
    smoking_history: formData.get("smoking_history"),
    bmi: formData.get("bmi"),
    HbA1c_level: formData.get("HbA1c_level"),
    blood_glucose_level: formData.get("blood_glucose_level"),
    diabetes: 0
  });

  if (record) {
    renderPrediction(predictProbability(encodeRow(record), appState.model));
  }
});

loadInitialDataset();
