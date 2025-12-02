import io

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats as scipy_stats


# ----------------------------
# Streamlit page configuration
# ----------------------------
st.set_page_config(
    page_title="Cobb–Douglas Production Function Explorer",
    layout="wide",
)

st.title("Cobb–Douglas Production Function: Employment Offices in Sweden")

st.markdown(
    """
This app lets you upload the Excel dataset from your assignment and automatically:

1. Clean and transform the data  
2. Create log variables for a Cobb–Douglas model  
3. Show descriptive statistics  
4. Draw scatter plots with fitted lines  
5. Estimate the Cobb–Douglas regression (OLS + robust SE)  
6. Run heteroskedasticity and normality tests  
7. Provide a downloadable text report for your write-up  
"""
)


# ----------------------------
# File upload
# ----------------------------
REQUIRED_COLS = [
    "F1",       # output
    "KVM",
    "ANT_FOR1",
    "ANT_VAG1",
    "U2_KONT",
    "U2_LA",
    "U2_KOM",
    "NYAVAK",
    "BEFOLKNI",
]

uploaded_file = st.file_uploader("Upload your Excel dataset", type=["xlsx", "xls"])

if uploaded_file is None:
    st.info(
        "⬆️ Upload the assignment Excel file to begin. "
        "It must contain these columns: "
        + ", ".join(REQUIRED_COLS)
    )
    st.stop()

# Try to read the Excel file
try:
    raw_df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Error reading Excel file: {e}")
    st.stop()

st.subheader("Raw Data Preview")
st.dataframe(raw_df.head())

missing_cols = [c for c in REQUIRED_COLS if c not in raw_df.columns]
if missing_cols:
    st.error(f"These required columns are missing in the file: {missing_cols}")
    st.stop()

# ----------------------------
# Data cleaning & log transform
# ----------------------------
st.header("1. Data Cleaning & Log Transformations")

# Keep only the required columns
df = raw_df[REQUIRED_COLS].copy()

# Convert to numeric where possible
for col in REQUIRED_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

st.write("**Missing values per column (before cleaning):**")
st.write(df.isna().sum())

# Drop rows with any missing values in the required columns
before_drop = len(df)
df = df.dropna()
after_drop_na = len(df)

# Drop rows with non-positive values (can’t take log)
df = df[(df > 0).all(axis=1)]
after_drop_nonpos = len(df)

st.markdown(
    f"""
- Rows before cleaning: **{before_drop}**  
- After dropping missing values: **{after_drop_na}**  
- After dropping non-positive values: **{after_drop_nonpos}**  
"""
)

if len(df) < 10:
    st.warning(
        "⚠️ Very few observations left after cleaning. "
        "Check your data or cleaning rules if this seems strange."
    )

# Create log variables
log_df = np.log(df)
log_df.columns = [f"ln_{c}" for c in df.columns]

st.subheader("Cleaned & Log-Transformed Data (first 5 rows)")
st.dataframe(pd.concat([df, log_df], axis=1).head())

# ----------------------------
# Descriptive statistics
# ----------------------------
st.header("2. Descriptive Statistics")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Original scale (levels)**")
    st.dataframe(df.describe().T)

with col2:
    st.markdown("**Log-transformed variables**")
    desc_log = log_df.describe().T
    st.dataframe(desc_log)

csv_log = desc_log.to_csv().encode("utf-8")
st.download_button(
    "📥 Download descriptive statistics (logs) as CSV",
    data=csv_log,
    file_name="descriptive_stats_log.csv",
    mime="text/csv",
)

# ----------------------------
# Scatter plots with fitted lines
# ----------------------------
st.header("3. Linearity Checks: ln(F1) vs ln(inputs)")

y = log_df["ln_F1"]
plot_vars = ["ln_KVM", "ln_ANT_FOR1", "ln_ANT_VAG1"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, var in zip(axes, plot_vars):
    x = log_df[var]

    # scatter
    ax.scatter(x, y, alpha=0.6)

    # simple linear regression for the line
    X_plot = sm.add_constant(x)
    model_plot = sm.OLS(y, X_plot).fit()
    x_line = np.linspace(x.min(), x.max(), 100)
    X_line = sm.add_constant(x_line)
    y_line = model_plot.predict(X_line)

    ax.plot(x_line, y_line, linewidth=2)
    ax.set_xlabel(var)
    ax.set_ylabel("ln_F1")
    ax.set_title(f"ln_F1 vs {var}")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# ----------------------------
# Cobb–Douglas regression
# ----------------------------
st.header("4. Cobb–Douglas Regression")

X_cols = [
    "ln_KVM",
    "ln_ANT_FOR1",
    "ln_ANT_VAG1",
    "ln_U2_KONT",
    "ln_U2_LA",
    "ln_U2_KOM",
    "ln_NYAVAK",
    "ln_BEFOLKNI",
]

# Align log_df cols
X = log_df[X_cols]
X = sm.add_constant(X)
y = log_df["ln_F1"]

model_ols = sm.OLS(y, X).fit()
model_robust = sm.OLS(y, X).fit(cov_type="HC1")

tab1, tab2 = st.tabs(["Classical OLS results", "OLS with robust (HC1) SE"])

with tab1:
    st.text(model_ols.summary())

with tab2:
    st.text(model_robust.summary())

# Some quick metrics
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("R-squared", f"{model_ols.rsquared:.3f}")
with c2:
    st.metric("Adj. R-squared", f"{model_ols.rsquared_adj:.3f}")
with c3:
    st.metric("Number of observations", f"{int(model_ols.nobs)}")

st.markdown(
    """
**Interpretation hints (for your assignment):**

- Coefficients on `ln_KVM`, `ln_ANT_FOR1`, `ln_ANT_VAG1` are **output elasticities**.  
  Example: if the coefficient on `ln_KVM` is 0.3, a 1% increase in office space  
  is associated with a ~0.3% increase in employment, ceteris paribus.  

- Control variables (`ln_U2_*`, `ln_NYAVAK`, `ln_BEFOLKNI`) capture labor-market
  and regional conditions to reduce omitted variable bias.
"""
)

# Returns to scale (only production inputs, not controls)
elasticities_inputs = [
    "ln_KVM",
    "ln_ANT_FOR1",
    "ln_ANT_VAG1",
]
rts_sum = sum(model_ols.params[v] for v in elasticities_inputs)

st.subheader("Returns to Scale (sum of input elasticities)")
st.metric("Sum of elasticities", f"{rts_sum:.3f}")

if rts_sum < 1:
    st.info("📉 **Decreasing returns to scale**: doubling inputs increases output by less than double.")
elif np.isclose(rts_sum, 1, atol=0.05):
    st.info("⚖️ **Approximately constant returns to scale**: doubling inputs roughly doubles output.")
else:
    st.info("📈 **Increasing returns to scale**: doubling inputs increases output by more than double.")

# ----------------------------
# Diagnostics: heteroskedasticity, residuals, normality
# ----------------------------
st.header("5. Diagnostics")

# Breusch–Pagan test
st.subheader("5.1 Breusch–Pagan Test for Heteroskedasticity")

bp_test = het_breuschpagan(model_ols.resid, X)
bp_labels = ["LM statistic", "LM p-value", "F statistic", "F p-value"]
bp_results = pd.DataFrame(
    [bp_test],
    columns=bp_labels,
)

st.dataframe(bp_results.style.format("{:.4f}"))

bp_pval = bp_results.loc[0, "LM p-value"]

if bp_pval < 0.05:
    st.warning(
        f"⚠️ Evidence of heteroskedasticity (LM p-value = {bp_pval:.4f} < 0.05). "
        "Robust standard errors (HC1) are more reliable."
    )
else:
    st.success(
        f"✅ No strong evidence of heteroskedasticity (LM p-value = {bp_pval:.4f} ≥ 0.05)."
    )

# Residual plots
st.subheader("5.2 Residual Diagnostics")

resid = model_ols.resid
fitted = model_ols.fittedvalues

col_res1, col_res2 = st.columns(2)

with col_res1:
    fig_res, ax_res = plt.subplots()
    ax_res.scatter(fitted, resid, alpha=0.6)
    ax_res.axhline(0, linestyle="--")
    ax_res.set_xlabel("Fitted values")
    ax_res.set_ylabel("Residuals")
    ax_res.set_title("Residuals vs Fitted")
    ax_res.grid(True, alpha=0.3)
    st.pyplot(fig_res)

with col_res2:
    fig_hist, ax_hist = plt.subplots()
    ax_hist.hist(resid, bins=20, edgecolor="black", alpha=0.7)
    ax_hist.set_title("Histogram of residuals")
    ax_hist.set_xlabel("Residual")
    ax_hist.set_ylabel("Frequency")
    ax_hist.grid(True, alpha=0.3)
    st.pyplot(fig_hist)

# Jarque–Bera normality test
st.subheader("5.3 Normality Test (Jarque–Bera)")

jb_stat, jb_pval = scipy_stats.jarque_bera(resid)
col_jb1, col_jb2 = st.columns(2)
with col_jb1:
    st.metric("Jarque–Bera statistic", f"{jb_stat:.3f}")
with col_jb2:
    st.metric("p-value", f"{jb_pval:.4f}")

if jb_pval < 0.05:
    st.warning(
        "⚠️ Residuals may **not** be normally distributed (p < 0.05). "
        "This mostly affects small-sample inference."
    )
else:
    st.success(
        "✅ Residuals are approximately normally distributed (p ≥ 0.05)."
    )

# ----------------------------
# Downloadable text report
# ----------------------------
st.header("6. Downloadable Summary for Your Report")

buffer = io.StringIO()
buffer.write("=== COBB–DOUGLAS PRODUCTION FUNCTION ANALYSIS ===\n\n")
buffer.write(f"Dataset: {len(df)} observations after cleaning\n\n")
buffer.write("=== CLASSICAL OLS RESULTS ===\n")
buffer.write(model_ols.summary().as_text())
buffer.write("\n\n=== ROBUST (HC1) RESULTS ===\n")
buffer.write(model_robust.summary().as_text())
buffer.write("\n\n=== RETURNS TO SCALE ===\n")
buffer.write(f"Sum of input elasticities (KVM, ANT_FOR1, ANT_VAG1): {rts_sum:.4f}\n")
buffer.write("\n=== BREUSCH–PAGAN TEST ===\n")
for label, value in zip(bp_labels, bp_test):
    buffer.write(f"{label}: {value:.4f}\n")
buffer.write("\n=== JARQUE–BERA TEST ===\n")
buffer.write(f"JB statistic: {jb_stat:.4f}, p-value: {jb_pval:.4f}\n")

report_text = buffer.getvalue()

st.download_button(
    "📥 Download full analysis report (text)",
    data=report_text,
    file_name="cobb_douglas_analysis.txt",
    mime="text/plain",
)

st.markdown(
    """
You can now use the outputs and this report to write up the **Method**, **Results**,  
**Diagnostics**, and **Conclusion** sections of your assignment.
"""
)
