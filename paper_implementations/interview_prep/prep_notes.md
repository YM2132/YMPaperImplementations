# Python Data Manipulation — Study Notes
### Quick Reference for Data Analysis & Modelling

---

## Contents
- [Data Cleaning](#data-cleaning)
- [Pandas Core Operations](#pandas-core-operations)
- [GroupBy Patterns](#groupby-patterns)
- [Merging & Reshaping](#merging--reshaping)
- [Linear Regression (sklearn)](#linear-regression-sklearn)
- [Residual Diagnostics](#residual-diagnostics)
- [Interpreting Regression Output](#interpreting-regression-output)
- [Finance Context](#finance-context)

---

## Data Cleaning

### Audit First — Always Start Here
```python
df.shape                  
df.dtypes                        
df.head(10)                       
df.describe()                     
df.isnull().sum()                 
df.duplicated().sum()             
df['col'].nunique()               
```

### Fix Strings
```python
df['ticker'] = df['ticker'].str.strip()      # remove leading/trailing whitespace
df['ticker'] = df['ticker'].str.upper()       # standardise case
df['ticker'] = df['ticker'].str.strip().str.upper()  # chain them

# Standardise inconsistent categories
smap = {'Tech': 'Technology', 'tech': 'Technology', 'TECH': 'Technology',
        'Healthcare': 'Healthcare', 'healthcare': 'Healthcare'}
df['sector'] = df['sector'].map(smap)
```

### Fix Numeric Columns (Strings Disguised as Numbers)
```python
# errors='coerce' turns unparseable values (#N/A, ERR, '', etc.) into NaN
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

# Check how many became NaN
df['price'].isna().sum()
```

### Fix Dates
```python
# format='mixed' handles multiple date formats in one column
# dayfirst=True assumes DD/MM/YYYY for ambiguous dates
df['date'] = pd.to_datetime(df['date'], dayfirst=True, format='mixed')
```

### Remove Duplicates
```python
df = df.drop_duplicates()                                    # exact duplicate rows
df = df.drop_duplicates(subset=['date', 'ticker'], keep='first')  # business key dupes
```

### Handle Missing Values
```python
df = df.dropna(subset=['critical_col1', 'critical_col2'])    # drop rows where key fields null
df['col'] = df['col'].fillna(0)                              # fill with value
df['col'] = df['col'].fillna(df['col'].median())             # fill with median
df['price'] = df['price'].ffill()                            # forward fill (last known value)
```

### Detect Outliers
```python
# Check describe for suspicious values
df['returns'].describe()

# Z-score method
z = (df['returns'] - df['returns'].mean()) / df['returns'].std()
outliers = df.loc[z.abs() > 3]

# Simple threshold (for returns: > 50% in a month is likely an error)
outliers = df.loc[df['returns'].abs() > 0.5]

# Winsorise (cap at percentiles)
lower, upper = df['returns'].quantile([0.01, 0.99])
df['returns_clean'] = df['returns'].clip(lower, upper)
```

---

## Pandas Core Operations

### Selecting & Filtering
```python
df['col']                             
df[['col1', 'col2']]                  
df.loc[df['sector'] == 'Tech']        
df.loc[df['weight'] > 0]              
df.loc[df['weight'] < 0]              
df.loc[df['ticker'].isin(['AAPL', 'MSFT'])] 

# Multiple conditions (use & for AND, | for OR, wrap each in parentheses)
df.loc[(df['sector'] == 'Tech') & (df['weight'] > 0)]
```

### Sorting
```python
df.sort_values('returns', ascending=False)              
df.sort_values(['sector', 'returns'], ascending=[True, False])  
df.nlargest(5, 'weight')                                
df.nsmallest(5, 'weight')                               
```

### Creating New Columns
```python
df['pnl'] = df['weight'] * df['returns']

# Conditional — two outcomes
df['side'] = np.where(df['weight'] > 0, 'Long', 'Short')

# Conditional — multiple outcomes
conditions = [df['market_cap'] > 100e9, df['market_cap'] > 10e9]
choices = ['Large', 'Mid']
df['size'] = np.select(conditions, choices, default='Small')

# From a mapping
side_map = {1: 'Long', -1: 'Short'}
df['side_label'] = df['side_code'].map(side_map)

# Log transform (useful for skewed data like market cap)
df['log_mcap'] = np.log(df['market_cap'])
```

### Ranking
```python
df['rank'] = df['returns'].rank(ascending=False)                      # overall rank
df['sector_rank'] = df.groupby('sector')['returns'].rank(ascending=False)  # rank within group
df['percentile'] = df.groupby('sector')['returns'].rank(pct=True)     # 0-1 percentile
```

### Value Counts & Unique
```python
df['sector'].value_counts()           # count per category
df['sector'].nunique()                # number of unique values
df['sector'].unique()                 # array of unique values
```

---

## GroupBy Patterns

### Basic Aggregation
```python
df.groupby('sector')['weight'].sum()           # sum per group
df.groupby('sector')['weight'].mean()          # mean per group
df.groupby('sector')['ticker'].count()         # count per group
```

### Named Aggregation (Multiple Metrics at Once)
```python
df.groupby('sector').agg(
    net_exposure=('weight', 'sum'),
    gross_exposure=('weight', lambda x: x.abs().sum()),
    n_long=('side', lambda x: (x == 'Long').sum()),
    n_short=('side', lambda x: (x == 'Short').sum()),
    count=('ticker', 'count')
).sort_values('gross_exposure', ascending=False)
```

### GroupBy with Lambda
```python
# Any custom function works inside lambda
df.groupby('sector')['weight'].apply(lambda x: x.abs().sum())
df.groupby('sector')['returns'].apply(lambda x: (x > 0).mean())  # % positive
```

### Multiple Group Keys
```python
df.groupby(['sector', 'side'])['weight'].sum()
df.groupby(['sector', 'side']).agg(count=('ticker', 'count'), total=('weight', 'sum'))
```

---

## Merging & Reshaping

### Merge (SQL-style Join)
```python
# Left join — keep all rows from left, add matching from right
merged = pd.merge(positions, prices, on=['date', 'ticker'], how='left')

# With indicator to see what matched
merged = pd.merge(positions, prices, on=['date', 'ticker'], how='left', indicator=True)
# _merge column: 'both' = matched, 'left_only' = no match in right

# Outer join — keep everything from both sides
full = pd.merge(positions, prices, on=['date', 'ticker'], how='outer', indicator=True)

# ALWAYS check row count after merge
print(f'Left: {len(positions)}, Right: {len(prices)}, Merged: {len(merged)}')
```

### Finding Discrepancies Between DataFrames
```python
set_a = set(df_a['ticker'].unique())
set_b = set(df_b['ticker'].unique())
only_in_a = set_a - set_b
only_in_b = set_b - set_a
in_both = set_a & set_b
```

### Concat (Stack DataFrames)
```python
combined = pd.concat([df1, df2], ignore_index=True)        # vertical (add rows)
combined = pd.concat([df1, df2], axis=1)                    # horizontal (add columns)
```

---

## Linear Regression (sklearn)

### Simple Regression (One Predictor)
```python
from sklearn.linear_model import LinearRegression

X = df[['predictor']]          # MUST be 2D — double brackets
y = df['target']

model = LinearRegression()
model.fit(X, y)

print(f'Intercept:   {model.intercept_:.4f}')
print(f'Coefficient: {model.coef_[0]:.4f}')
print(f'R-squared:   {model.score(X, y):.4f}')

y_pred = model.predict(X)
residuals = y - y_pred
```

### Multiple Regression (Several Predictors)
```python
X = df[['pred1', 'pred2', 'pred3', 'pred4']]
y = df['target']

model = LinearRegression()
model.fit(X, y)

print(f'R-squared: {model.score(X, y):.4f}')
print(f'Intercept: {model.intercept_:.6f}')
for name, coef in zip(X.columns, model.coef_):
    print(f'  {name}: {coef:.6f}')

y_pred = model.predict(X)
residuals = y - y_pred
```

### Scatter Plot with Regression Line (Simple Regression)
```python
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Predictor')
plt.ylabel('Target')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()
```

### Dual-Axis Chart (Two Series, Different Scales)
```python
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(df['x'], df['y1'], color='red', label='Series 1')
ax1.set_ylabel('Y1 Label', color='red')
ax1.tick_params(axis='y', labelcolor='red')

ax2 = ax1.twinx()
ax2.plot(df['x'], df['y2'], color='blue', alpha=0.5, label='Series 2')
ax2.set_ylabel('Y2 Label', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

plt.title('Title')
fig.tight_layout()
plt.show()
```

### Two Series, Same Scale
```python
plt.plot(df['x'], df['y1'], label='Series 1')
plt.plot(df['x'], df['y2'], label='Series 2', linestyle='--')
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Title')
plt.legend()
plt.show()
```

---

## Residual Diagnostics

### Regression plots
```python
from scipy import stats
import matplotlib.pyplot as plt

# Create 3-panel diagnostic plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Residuals vs Fitted (Checks Linearity & Homoscedasticity)
axes[0].scatter(y_pred, residuals, alpha=0.5)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('Fitted Values')
axes[0].set_ylabel('Residuals')
axes[0].set_title('1. Residuals vs Fitted')

# 2. Histogram (Checks Normality)
axes[1].hist(residuals, bins=25, edgecolor='black', density=True)
x_norm = np.linspace(residuals.min(), residuals.max(), 100)
axes[1].plot(x_norm, stats.norm.pdf(x_norm, residuals.mean(), residuals.std()), 'r-', lw=2)
axes[1].set_title('2. Residual Distribution')
axes[1].set_xlabel('Residual')

# 3. QQ Plot (Checks Normality in Tails)
stats.probplot(residuals, plot=axes[2])
axes[2].set_title('3. QQ Plot')

plt.tight_layout()
plt.show()
```

### Shapiro-Wilk Normality Test
```python
from scipy import stats

stat, p = stats.shapiro(residuals)
print(f'Shapiro-Wilk: stat={stat:.4f}, p={p:.4f}')
# p > 0.05 -> residuals are approximately normal
# p < 0.05 -> residuals may not be normal
```

### Multicollinearity Check (Multiple Regression Only)
```python
# Correlation matrix between predictors
print(df[['pred1', 'pred2', 'pred3']].corr().round(3))
# |correlation| > 0.7 between two predictors = concerning
```

---

## Interpreting Regression Output

### R-Squared
- Ranges from 0 to 1
- "The model explains X% of the variance in [target]"
- In finance, R-squared of 0.02–0.05 is normal for return prediction
- Low R-squared doesn't mean the model is useless — returns are inherently noisy

### Coefficients
- Each coefficient: "for a 1-unit increase in [predictor], [target] changes by [coef], holding all other predictors constant"
- Check direction: does positive/negative make economic sense?
- For volatility (range 0.10–0.60): a "1-unit increase" is unrealistic. Better to say "a 10 percentage point increase (0.10) is associated with [coef × 0.10] change in return"

## Regression Assumption Checklist

1. Linearity         → residuals vs fitted: no curve
   plt.scatter(y_pred, residuals); plt.axhline(y=0, color='r', linestyle='--')

2. Independence      → residuals over time: no tracking
   plt.plot(residuals); plt.axhline(y=0, color='r', linestyle='--')

3. Homoscedasticity  → residuals vs fitted: no funnel
   # same plot as #1 — funnel shape = variance not constant

4. Normality         → histogram + QQ plot
   plt.hist(residuals, bins=25)
   from scipy import stats; stats.probplot(residuals, plot=plt)

5. Multicollinearity → predictor correlation matrix (multiple regression only)
   df[['pred1', 'pred2']].corr()   # |corr| > 0.7 = concerning

6. Outliers/Leverage → z-score > |3|
   z = (df['col'] - df['col'].mean()) / df['col'].std()
   df[z.abs() > 3]
---

## Finance Context

### Portfolio Exposure
```python
net_exposure = df['weight'].sum()            # long - short (directional bet)
gross_exposure = df['weight'].abs().sum()     # total capital deployed

# In L/S funds, weights DON'T sum to 1 (leverage)
# e.g. 90% long + 70% short = 160% gross, 20% net
```

### Long vs Short
- Long weights are positive, short weights are negative
- The negative sign handles directionality automatically
- weight × return: negative × negative = positive (short makes money when stock falls)

### Returns
```python
# Simple return
df['return'] = df['price'].pct_change()

# Cumulative return (compounding)
cumulative = (1 + df['return']).cumprod() - 1

# Annualise from daily
ann_return = daily_return.mean() * 252
ann_vol = daily_return.std() * np.sqrt(252)
sharpe = (ann_return - risk_free_rate) / ann_vol
```
---