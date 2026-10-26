# Phase 1 EDA snippets

Historical scratchpad from Phase 1. These fragments are preserved for reference and are not a standalone script.

```text
# Temporal patterns (order volume, seasonality)
- Overall order volume trend
- Seasonality at the monthly level
- Seasonality at the day-of-week level
- Order Status over Time (To explain end of dataset drop to 0)
- Month-over-Month Growth (%) (Growth rate quantification)


# Monthly Order Volume
monthly_orders = df.groupby('year_month')['order_id'].nunique()

fig_monthly_orders = plt.figure(figsize=(12, 6))
plt.plot(monthly_orders.index.astype(str), monthly_orders.values, marker='o', linestyle='-', color='#2E86AB')
plt.title('Monthly Order Volume', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Number of Unique Orders', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Orders by Day of Week
day_of_week_order_counts = df.groupby('day_of_week')['order_id'].nunique().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])

fig_day_of_week = plt.figure(figsize=(12, 6))
sns.barplot(x=day_of_week_order_counts.index, y=day_of_week_order_counts.values, palette='viridis')
plt.title('Number of Unique Orders by Day of Week', fontsize=16, fontweight='bold')
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Number of Unique Orders', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Order Status Trend over Time
order_status_over_time = df.groupby(['year_month', 'order_status'])['order_id'].nunique().unstack(fill_value=0)

order_status_over_time.plot(figsize=(12, 6), marker='o')
plt.title('Order Status Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# Rolling Average / Growth Rate (Month-over-Month %)  [Line]
monthly_orders = df.groupby('year_month')['order_id'].nunique()
monthly_growth = monthly_orders.pct_change() * 100

plt.figure(figsize=(12, 6))
monthly_growth.plot(kind='line', marker='o', color='#2E86AB')
plt.title('Overall Month-over-Month Order Growth Rate (%)', fontweight='bold', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Growth Rate (%)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Rolling Average / Growth Rate (Month-over-Month %)  [Bar]
monthly_orders = df.groupby('year_month')['order_id'].nunique()
monthly_growth = monthly_orders.pct_change() * 100

plt.figure(figsize=(12, 6))
monthly_growth.plot(kind='bar', color='#F18F01')
plt.title('Month-over-Month Order Growth (%)', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Growth Rate (%)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```
