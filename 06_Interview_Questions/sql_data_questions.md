# SQL & Data Interview Questions

## Table of Contents
1. [SQL Fundamentals](#sql-fundamentals)
2. [Joins and Subqueries](#joins-and-subqueries)
3. [Window Functions](#window-functions)
4. [Aggregations and Grouping](#aggregations-and-grouping)
5. [Data Manipulation](#data-manipulation)
6. [Performance and Optimization](#performance-and-optimization)
7. [Practical Problems](#practical-problems)

---

## SQL Fundamentals

### Q1: Order of SQL execution
```sql
-- Written order vs Execution order

-- Written:
SELECT, FROM, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT

-- Execution:
1. FROM / JOIN      -- Get data from tables
2. WHERE            -- Filter rows
3. GROUP BY         -- Aggregate rows
4. HAVING           -- Filter groups
5. SELECT           -- Select columns
6. DISTINCT         -- Remove duplicates
7. ORDER BY         -- Sort results
8. LIMIT / OFFSET   -- Limit results

-- This is why you can't use SELECT aliases in WHERE
-- But can use them in ORDER BY
SELECT name, salary * 12 AS annual_salary
FROM employees
WHERE salary > 50000           -- Can't use annual_salary here
ORDER BY annual_salary;        -- Can use alias here
```

### Q2: INNER JOIN vs LEFT JOIN vs FULL OUTER JOIN
```sql
-- Sample tables
-- users: id, name
-- orders: id, user_id, amount

-- INNER JOIN: Only matching rows from both tables
SELECT u.name, o.amount
FROM users u
INNER JOIN orders o ON u.id = o.user_id;

-- LEFT JOIN: All rows from left + matching from right
-- NULL for non-matching right rows
SELECT u.name, o.amount
FROM users u
LEFT JOIN orders o ON u.id = o.user_id;

-- RIGHT JOIN: All rows from right + matching from left
SELECT u.name, o.amount
FROM users u
RIGHT JOIN orders o ON u.id = o.user_id;

-- FULL OUTER JOIN: All rows from both tables
-- NULL where no match
SELECT u.name, o.amount
FROM users u
FULL OUTER JOIN orders o ON u.id = o.user_id;

-- CROSS JOIN: Cartesian product (all combinations)
SELECT u.name, p.product_name
FROM users u
CROSS JOIN products p;
```

### Q3: UNION vs UNION ALL vs INTERSECT vs EXCEPT
```sql
-- UNION: Combines results, removes duplicates
SELECT city FROM customers
UNION
SELECT city FROM suppliers;

-- UNION ALL: Combines results, keeps duplicates (faster)
SELECT city FROM customers
UNION ALL
SELECT city FROM suppliers;

-- INTERSECT: Rows that appear in both queries
SELECT city FROM customers
INTERSECT
SELECT city FROM suppliers;

-- EXCEPT: Rows in first query but not in second
SELECT city FROM customers
EXCEPT
SELECT city FROM suppliers;

-- Note: Column count and types must match
```

### Q4: NULL handling
```sql
-- NULL comparisons always return NULL (not TRUE/FALSE)
SELECT * FROM users WHERE age = NULL;      -- Returns nothing!
SELECT * FROM users WHERE age IS NULL;     -- Correct

-- NULL in expressions
SELECT 5 + NULL;           -- NULL
SELECT COALESCE(NULL, 0);  -- 0 (first non-null)
SELECT NULLIF(5, 5);       -- NULL (returns NULL if equal)

-- NULL in aggregations
SELECT AVG(salary) FROM employees;  -- NULLs are ignored
SELECT COUNT(salary) FROM employees;  -- Counts non-NULL only
SELECT COUNT(*) FROM employees;       -- Counts all rows

-- NULL-safe equality (MySQL)
SELECT * FROM t1, t2 WHERE t1.x <=> t2.x;  -- Treats NULL=NULL as TRUE
```

---

## Joins and Subqueries

### Q5: Self-join examples
```sql
-- Find employees and their managers
SELECT 
    e.name AS employee,
    m.name AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id;

-- Find pairs of users from same city
SELECT 
    u1.name AS user1,
    u2.name AS user2,
    u1.city
FROM users u1
JOIN users u2 ON u1.city = u2.city AND u1.id < u2.id;

-- Find consecutive dates with events
SELECT 
    e1.date,
    e1.event_type
FROM events e1
JOIN events e2 ON e1.date = e2.date - INTERVAL '1 day'
                  AND e1.user_id = e2.user_id;
```

### Q6: Correlated vs Non-correlated subqueries
```sql
-- Non-correlated: Subquery runs once
-- Find employees earning above average
SELECT name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

-- Correlated: Subquery runs for each outer row
-- Find employees earning above their department average
SELECT e.name, e.salary, e.dept_id
FROM employees e
WHERE e.salary > (
    SELECT AVG(e2.salary) 
    FROM employees e2 
    WHERE e2.dept_id = e.dept_id
);

-- Correlated subquery with EXISTS
-- Find customers with at least one order
SELECT c.name
FROM customers c
WHERE EXISTS (
    SELECT 1 
    FROM orders o 
    WHERE o.customer_id = c.id
);

-- Often can rewrite as JOIN for better performance
SELECT DISTINCT c.name
FROM customers c
JOIN orders o ON o.customer_id = c.id;
```

### Q7: CTEs (Common Table Expressions)
```sql
-- Basic CTE
WITH active_users AS (
    SELECT user_id, COUNT(*) as login_count
    FROM logins
    WHERE login_date > CURRENT_DATE - INTERVAL '30 days'
    GROUP BY user_id
)
SELECT u.name, au.login_count
FROM users u
JOIN active_users au ON u.id = au.user_id;

-- Multiple CTEs
WITH 
monthly_sales AS (
    SELECT 
        DATE_TRUNC('month', order_date) AS month,
        SUM(amount) AS total
    FROM orders
    GROUP BY 1
),
sales_growth AS (
    SELECT 
        month,
        total,
        LAG(total) OVER (ORDER BY month) AS prev_total
    FROM monthly_sales
)
SELECT 
    month,
    total,
    ROUND((total - prev_total) / prev_total * 100, 2) AS growth_pct
FROM sales_growth;

-- Recursive CTE (for hierarchical data)
WITH RECURSIVE org_chart AS (
    -- Base case: top-level employees
    SELECT id, name, manager_id, 1 AS level
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    -- Recursive case
    SELECT e.id, e.name, e.manager_id, oc.level + 1
    FROM employees e
    JOIN org_chart oc ON e.manager_id = oc.id
)
SELECT * FROM org_chart ORDER BY level, name;
```

---

## Window Functions

### Q8: ROW_NUMBER, RANK, DENSE_RANK
```sql
-- Sample data: scores by student
-- ROW_NUMBER: Unique sequential numbers
-- RANK: Same rank for ties, gaps after
-- DENSE_RANK: Same rank for ties, no gaps

SELECT 
    student_name,
    score,
    ROW_NUMBER() OVER (ORDER BY score DESC) AS row_num,
    RANK() OVER (ORDER BY score DESC) AS rank,
    DENSE_RANK() OVER (ORDER BY score DESC) AS dense_rank
FROM exam_results;

-- Results for scores: 95, 90, 90, 85
-- ROW_NUMBER: 1, 2, 3, 4
-- RANK:       1, 2, 2, 4
-- DENSE_RANK: 1, 2, 2, 3

-- Top N per group
WITH ranked AS (
    SELECT 
        department,
        employee_name,
        salary,
        ROW_NUMBER() OVER (
            PARTITION BY department 
            ORDER BY salary DESC
        ) AS rn
    FROM employees
)
SELECT * FROM ranked WHERE rn <= 3;
```

### Q9: LAG and LEAD functions
```sql
-- LAG: Access previous row
-- LEAD: Access next row

SELECT 
    date,
    stock_price,
    LAG(stock_price, 1) OVER (ORDER BY date) AS prev_price,
    LEAD(stock_price, 1) OVER (ORDER BY date) AS next_price,
    stock_price - LAG(stock_price) OVER (ORDER BY date) AS daily_change
FROM stock_prices;

-- With default value for NULL
SELECT 
    date,
    value,
    LAG(value, 1, 0) OVER (ORDER BY date) AS prev_value
FROM metrics;

-- Calculate day-over-day growth rate
SELECT 
    date,
    revenue,
    ROUND(
        (revenue - LAG(revenue) OVER (ORDER BY date)) 
        / LAG(revenue) OVER (ORDER BY date) * 100, 
        2
    ) AS growth_rate
FROM daily_revenue;
```

### Q10: Running totals and moving averages
```sql
-- Running total
SELECT 
    date,
    amount,
    SUM(amount) OVER (ORDER BY date) AS running_total
FROM transactions;

-- Running total partitioned by category
SELECT 
    category,
    date,
    amount,
    SUM(amount) OVER (
        PARTITION BY category 
        ORDER BY date
    ) AS category_running_total
FROM transactions;

-- Moving average (last 7 days)
SELECT 
    date,
    value,
    AVG(value) OVER (
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS moving_avg_7d
FROM metrics;

-- Moving average with RANGE (handles gaps)
SELECT 
    date,
    value,
    AVG(value) OVER (
        ORDER BY date
        RANGE BETWEEN INTERVAL '7 days' PRECEDING AND CURRENT ROW
    ) AS moving_avg_7d
FROM metrics;
```

### Q11: FIRST_VALUE, LAST_VALUE, NTH_VALUE
```sql
-- FIRST_VALUE: First value in window
SELECT 
    employee_name,
    department,
    salary,
    FIRST_VALUE(employee_name) OVER (
        PARTITION BY department 
        ORDER BY salary DESC
    ) AS highest_paid_in_dept
FROM employees;

-- LAST_VALUE: Last value in window
-- Note: Default frame is RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
-- Need to specify full frame for expected behavior
SELECT 
    employee_name,
    department,
    salary,
    LAST_VALUE(employee_name) OVER (
        PARTITION BY department 
        ORDER BY salary DESC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS lowest_paid_in_dept
FROM employees;

-- NTH_VALUE: Nth value in window
SELECT 
    employee_name,
    department,
    salary,
    NTH_VALUE(employee_name, 2) OVER (
        PARTITION BY department 
        ORDER BY salary DESC
    ) AS second_highest_paid
FROM employees;
```

---

## Aggregations and Grouping

### Q12: GROUP BY with ROLLUP, CUBE, GROUPING SETS
```sql
-- ROLLUP: Hierarchical subtotals
SELECT 
    region,
    product,
    SUM(sales) AS total_sales
FROM sales_data
GROUP BY ROLLUP(region, product);
-- Gives: (region, product), (region, NULL), (NULL, NULL)

-- CUBE: All possible subtotals
SELECT 
    region,
    product,
    SUM(sales) AS total_sales
FROM sales_data
GROUP BY CUBE(region, product);
-- Gives all combinations including (NULL, product)

-- GROUPING SETS: Specific subtotals
SELECT 
    region,
    product,
    SUM(sales) AS total_sales
FROM sales_data
GROUP BY GROUPING SETS (
    (region, product),
    (region),
    (product),
    ()
);

-- GROUPING function to identify subtotal rows
SELECT 
    region,
    product,
    SUM(sales) AS total_sales,
    GROUPING(region) AS is_region_total,
    GROUPING(product) AS is_product_total
FROM sales_data
GROUP BY ROLLUP(region, product);
```

### Q13: HAVING vs WHERE
```sql
-- WHERE: Filters rows BEFORE grouping
-- HAVING: Filters groups AFTER aggregation

-- Find departments with average salary > 50000
SELECT 
    department,
    AVG(salary) AS avg_salary
FROM employees
WHERE status = 'active'      -- Filter rows first
GROUP BY department
HAVING AVG(salary) > 50000;  -- Then filter groups

-- Common mistake: Using WHERE with aggregates
-- This is WRONG:
-- SELECT department FROM employees WHERE AVG(salary) > 50000

-- Multiple HAVING conditions
SELECT 
    department,
    COUNT(*) AS emp_count,
    AVG(salary) AS avg_salary
FROM employees
GROUP BY department
HAVING COUNT(*) >= 5 AND AVG(salary) > 40000;
```

---

## Data Manipulation

### Q14: INSERT, UPDATE, DELETE patterns
```sql
-- INSERT with SELECT
INSERT INTO archive_orders (order_id, customer_id, amount)
SELECT order_id, customer_id, amount
FROM orders
WHERE order_date < '2023-01-01';

-- UPDATE with JOIN
UPDATE orders o
SET status = 'premium'
FROM customers c
WHERE o.customer_id = c.id AND c.tier = 'gold';

-- UPDATE with subquery
UPDATE products
SET price = price * 1.1
WHERE category_id IN (
    SELECT id FROM categories WHERE name = 'Electronics'
);

-- DELETE with JOIN
DELETE FROM orders
USING customers
WHERE orders.customer_id = customers.id
  AND customers.status = 'inactive';

-- UPSERT (INSERT or UPDATE)
INSERT INTO user_stats (user_id, login_count)
VALUES (1, 1)
ON CONFLICT (user_id) 
DO UPDATE SET login_count = user_stats.login_count + 1;
```

### Q15: CASE expressions
```sql
-- Simple CASE
SELECT 
    order_id,
    amount,
    CASE 
        WHEN amount < 100 THEN 'Small'
        WHEN amount < 500 THEN 'Medium'
        WHEN amount < 1000 THEN 'Large'
        ELSE 'Enterprise'
    END AS order_size
FROM orders;

-- CASE in aggregation (pivot-like)
SELECT 
    product_id,
    SUM(CASE WHEN month = 1 THEN revenue ELSE 0 END) AS jan_revenue,
    SUM(CASE WHEN month = 2 THEN revenue ELSE 0 END) AS feb_revenue,
    SUM(CASE WHEN month = 3 THEN revenue ELSE 0 END) AS mar_revenue
FROM monthly_sales
GROUP BY product_id;

-- CASE in ORDER BY
SELECT name, status
FROM tasks
ORDER BY 
    CASE status
        WHEN 'urgent' THEN 1
        WHEN 'high' THEN 2
        WHEN 'medium' THEN 3
        ELSE 4
    END;
```

---

## Performance and Optimization

### Q16: Index usage and query optimization
```sql
-- Indexes speed up: WHERE, JOIN, ORDER BY, GROUP BY

-- When indexes are NOT used:
-- 1. Functions on columns
SELECT * FROM users WHERE YEAR(created_at) = 2023;  -- No index
SELECT * FROM users WHERE created_at >= '2023-01-01' 
                      AND created_at < '2024-01-01';  -- Uses index

-- 2. Leading wildcard in LIKE
SELECT * FROM users WHERE name LIKE '%smith';  -- No index
SELECT * FROM users WHERE name LIKE 'smith%';  -- Uses index

-- 3. OR conditions (sometimes)
SELECT * FROM users WHERE city = 'NYC' OR state = 'CA';  -- May not use

-- 4. Implicit type conversion
SELECT * FROM orders WHERE order_id = '123';  -- If order_id is INT

-- Composite index order matters
-- Index on (a, b, c) can be used for:
-- WHERE a = 1
-- WHERE a = 1 AND b = 2
-- WHERE a = 1 AND b = 2 AND c = 3
-- But NOT for:
-- WHERE b = 2
-- WHERE b = 2 AND c = 3

-- EXPLAIN to analyze queries
EXPLAIN ANALYZE
SELECT * FROM orders WHERE customer_id = 100;
```

### Q17: Query optimization techniques
```sql
-- 1. Avoid SELECT *
SELECT id, name FROM users;  -- Better than SELECT *

-- 2. Use EXISTS instead of IN for large sets
-- Slow:
SELECT * FROM products 
WHERE category_id IN (SELECT id FROM categories WHERE active = true);

-- Faster:
SELECT * FROM products p
WHERE EXISTS (
    SELECT 1 FROM categories c 
    WHERE c.id = p.category_id AND c.active = true
);

-- 3. Limit early
SELECT * FROM large_table
WHERE created_at > '2023-01-01'
ORDER BY created_at
LIMIT 100;

-- 4. Use appropriate data types
-- INT is faster than VARCHAR for joins

-- 5. Avoid correlated subqueries when possible
-- Slow (correlated):
SELECT e.name, 
       (SELECT AVG(salary) FROM employees e2 WHERE e2.dept = e.dept)
FROM employees e;

-- Faster (join):
SELECT e.name, d.avg_salary
FROM employees e
JOIN (SELECT dept, AVG(salary) as avg_salary FROM employees GROUP BY dept) d
ON e.dept = d.dept;
```

---

## Practical Problems

### Q18: Find duplicate records
```sql
-- Find duplicate emails
SELECT email, COUNT(*) as count
FROM users
GROUP BY email
HAVING COUNT(*) > 1;

-- Get all rows with duplicate emails
SELECT *
FROM users
WHERE email IN (
    SELECT email
    FROM users
    GROUP BY email
    HAVING COUNT(*) > 1
);

-- Delete duplicates, keep one
DELETE FROM users
WHERE id NOT IN (
    SELECT MIN(id)
    FROM users
    GROUP BY email
);

-- Using window function
WITH duplicates AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY email ORDER BY id) AS rn
    FROM users
)
DELETE FROM users WHERE id IN (
    SELECT id FROM duplicates WHERE rn > 1
);
```

### Q19: Consecutive days/sessions problem
```sql
-- Find users with 3+ consecutive login days
WITH login_groups AS (
    SELECT 
        user_id,
        login_date,
        login_date - ROW_NUMBER() OVER (
            PARTITION BY user_id 
            ORDER BY login_date
        ) * INTERVAL '1 day' AS grp
    FROM (SELECT DISTINCT user_id, login_date FROM logins) t
)
SELECT user_id, COUNT(*) AS consecutive_days
FROM login_groups
GROUP BY user_id, grp
HAVING COUNT(*) >= 3;

-- Find longest streak per user
WITH login_groups AS (
    SELECT 
        user_id,
        login_date,
        login_date - ROW_NUMBER() OVER (
            PARTITION BY user_id 
            ORDER BY login_date
        ) * INTERVAL '1 day' AS grp
    FROM (SELECT DISTINCT user_id, login_date FROM logins) t
),
streaks AS (
    SELECT user_id, grp, COUNT(*) AS streak_length
    FROM login_groups
    GROUP BY user_id, grp
)
SELECT user_id, MAX(streak_length) AS longest_streak
FROM streaks
GROUP BY user_id;
```

### Q20: Year-over-year comparison
```sql
-- Monthly revenue with YoY comparison
WITH monthly_revenue AS (
    SELECT 
        DATE_TRUNC('month', order_date) AS month,
        SUM(amount) AS revenue
    FROM orders
    GROUP BY 1
)
SELECT 
    m1.month,
    m1.revenue AS current_revenue,
    m2.revenue AS previous_year_revenue,
    ROUND((m1.revenue - m2.revenue) / m2.revenue * 100, 2) AS yoy_growth
FROM monthly_revenue m1
LEFT JOIN monthly_revenue m2 
    ON m1.month = m2.month + INTERVAL '1 year';

-- Using LAG with offset of 12 months
SELECT 
    month,
    revenue,
    LAG(revenue, 12) OVER (ORDER BY month) AS prev_year_revenue,
    ROUND(
        (revenue - LAG(revenue, 12) OVER (ORDER BY month)) 
        / LAG(revenue, 12) OVER (ORDER BY month) * 100, 
        2
    ) AS yoy_growth
FROM monthly_revenue;
```

### Q21: Retention analysis
```sql
-- Calculate monthly cohort retention
WITH first_purchase AS (
    SELECT 
        user_id,
        DATE_TRUNC('month', MIN(order_date)) AS cohort_month
    FROM orders
    GROUP BY user_id
),
user_activity AS (
    SELECT 
        f.user_id,
        f.cohort_month,
        DATE_TRUNC('month', o.order_date) AS activity_month
    FROM first_purchase f
    JOIN orders o ON f.user_id = o.user_id
)
SELECT 
    cohort_month,
    COUNT(DISTINCT user_id) AS cohort_size,
    COUNT(DISTINCT CASE 
        WHEN activity_month = cohort_month + INTERVAL '1 month' 
        THEN user_id 
    END) AS month_1_retained,
    COUNT(DISTINCT CASE 
        WHEN activity_month = cohort_month + INTERVAL '2 month' 
        THEN user_id 
    END) AS month_2_retained
FROM user_activity
GROUP BY cohort_month
ORDER BY cohort_month;
```

### Q22: Percentile and median
```sql
-- Median salary (PostgreSQL)
SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) AS median_salary
FROM employees;

-- Median per department
SELECT 
    department,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) AS median_salary
FROM employees
GROUP BY department;

-- Multiple percentiles
SELECT 
    department,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY salary) AS p25,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY salary) AS p50,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY salary) AS p75
FROM employees
GROUP BY department;

-- Median without PERCENTILE_CONT (works in MySQL)
SELECT AVG(salary) AS median_salary
FROM (
    SELECT salary
    FROM employees
    ORDER BY salary
    LIMIT 2 - (SELECT COUNT(*) FROM employees) % 2
    OFFSET (SELECT (COUNT(*) - 1) / 2 FROM employees)
) sub;
```

### Q23: Gaps and islands problem
```sql
-- Find gaps in sequential IDs
WITH all_ids AS (
    SELECT generate_series(
        (SELECT MIN(id) FROM orders),
        (SELECT MAX(id) FROM orders)
    ) AS id
)
SELECT a.id AS missing_id
FROM all_ids a
LEFT JOIN orders o ON a.id = o.id
WHERE o.id IS NULL;

-- Find islands (consecutive groups)
WITH numbered AS (
    SELECT 
        id,
        id - ROW_NUMBER() OVER (ORDER BY id) AS grp
    FROM orders
)
SELECT 
    MIN(id) AS island_start,
    MAX(id) AS island_end,
    COUNT(*) AS island_size
FROM numbered
GROUP BY grp
ORDER BY island_start;
```

### Q24: Pivoting data
```sql
-- Manual pivot using CASE
SELECT 
    product_id,
    MAX(CASE WHEN attribute = 'color' THEN value END) AS color,
    MAX(CASE WHEN attribute = 'size' THEN value END) AS size,
    MAX(CASE WHEN attribute = 'weight' THEN value END) AS weight
FROM product_attributes
GROUP BY product_id;

-- PostgreSQL crosstab
SELECT * FROM crosstab(
    'SELECT product_id, attribute, value
     FROM product_attributes
     ORDER BY 1, 2'
) AS ct(product_id INT, color TEXT, size TEXT, weight TEXT);

-- Unpivot (turn columns to rows)
SELECT product_id, 'color' AS attribute, color AS value FROM products
UNION ALL
SELECT product_id, 'size' AS attribute, size AS value FROM products
UNION ALL
SELECT product_id, 'weight' AS attribute, weight AS value FROM products;
```

### Q25: Funnel analysis
```sql
-- E-commerce funnel: view → cart → checkout → purchase
WITH funnel AS (
    SELECT 
        user_id,
        MAX(CASE WHEN event = 'view' THEN 1 ELSE 0 END) AS viewed,
        MAX(CASE WHEN event = 'add_to_cart' THEN 1 ELSE 0 END) AS added_to_cart,
        MAX(CASE WHEN event = 'checkout' THEN 1 ELSE 0 END) AS checked_out,
        MAX(CASE WHEN event = 'purchase' THEN 1 ELSE 0 END) AS purchased
    FROM events
    WHERE event_date BETWEEN '2023-01-01' AND '2023-01-31'
    GROUP BY user_id
)
SELECT 
    SUM(viewed) AS views,
    SUM(added_to_cart) AS add_to_carts,
    SUM(checked_out) AS checkouts,
    SUM(purchased) AS purchases,
    ROUND(SUM(added_to_cart)::NUMERIC / SUM(viewed) * 100, 2) AS view_to_cart_rate,
    ROUND(SUM(checked_out)::NUMERIC / SUM(added_to_cart) * 100, 2) AS cart_to_checkout_rate,
    ROUND(SUM(purchased)::NUMERIC / SUM(checked_out) * 100, 2) AS checkout_to_purchase_rate
FROM funnel;
```
