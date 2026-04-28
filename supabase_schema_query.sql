-- ============================================================
-- SAYSAVOR — Full Database Schema (Safe Version)
-- Run ONE section at a time in Supabase SQL Editor
-- Each section is separated by a comment header
-- ============================================================


-- ════════════════════════════════════════════════════════════
-- SECTION 1: ALL TABLES + COLUMN COUNT + ESTIMATED ROW COUNT
-- ════════════════════════════════════════════════════════════
SELECT
    t.table_name                                   AS "Table",
    (SELECT COUNT(*)
     FROM information_schema.columns c
     WHERE c.table_schema = 'public'
       AND c.table_name   = t.table_name)          AS "Columns",
    pg_size_pretty(pg_total_relation_size(
        quote_ident(t.table_name)::regclass))      AS "Size",
    (SELECT reltuples::BIGINT
     FROM pg_class
     WHERE relname = t.table_name)                 AS "Est Rows"
FROM information_schema.tables t
WHERE t.table_schema = 'public'
  AND t.table_type   = 'BASE TABLE'
ORDER BY t.table_name;


-- ════════════════════════════════════════════════════════════
-- SECTION 2: EVERY COLUMN IN EVERY TABLE (full detail)
-- ════════════════════════════════════════════════════════════
SELECT
    c.table_name                                   AS "Table",
    c.ordinal_position                             AS "Pos",
    c.column_name                                  AS "Column",
    c.data_type                                    AS "Type",
    c.udt_name                                     AS "Subtype",
    c.character_maximum_length                     AS "MaxLen",
    c.is_nullable                                  AS "Nullable",
    c.column_default                               AS "Default",
    CASE
        WHEN pk.column_name IS NOT NULL THEN 'PRIMARY KEY'
        WHEN fk.column_name IS NOT NULL
             THEN 'FK → ' || fk.foreign_table || '.' || fk.foreign_col
        ELSE ''
    END                                            AS "Key / Relation"
FROM information_schema.columns c

LEFT JOIN (
    SELECT ku.table_name, ku.column_name
    FROM information_schema.table_constraints tc
    JOIN information_schema.key_column_usage ku
        ON tc.constraint_name = ku.constraint_name
       AND tc.table_schema    = ku.table_schema
    WHERE tc.constraint_type = 'PRIMARY KEY'
      AND tc.table_schema    = 'public'
) pk ON pk.table_name = c.table_name AND pk.column_name = c.column_name

LEFT JOIN (
    SELECT
        kcu.table_name,
        kcu.column_name,
        ccu.table_name  AS foreign_table,
        ccu.column_name AS foreign_col
    FROM information_schema.table_constraints tc
    JOIN information_schema.key_column_usage kcu
        ON tc.constraint_name = kcu.constraint_name
    JOIN information_schema.constraint_column_usage ccu
        ON ccu.constraint_name = tc.constraint_name
    WHERE tc.constraint_type = 'FOREIGN KEY'
      AND tc.table_schema    = 'public'
) fk ON fk.table_name = c.table_name AND fk.column_name = c.column_name

WHERE c.table_schema = 'public'
ORDER BY c.table_name, c.ordinal_position;


-- ════════════════════════════════════════════════════════════
-- SECTION 3: FOREIGN KEY RELATIONSHIPS (who links to whom)
-- ════════════════════════════════════════════════════════════
SELECT
    kcu.table_name              AS "Child Table",
    kcu.column_name             AS "Child Column",
    ccu.table_name              AS "Parent Table",
    ccu.column_name             AS "Parent Column",
    tc.constraint_name          AS "FK Constraint Name"
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu
    ON tc.constraint_name = kcu.constraint_name
   AND tc.table_schema    = kcu.table_schema
JOIN information_schema.constraint_column_usage ccu
    ON ccu.constraint_name = tc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY'
  AND tc.table_schema    = 'public'
ORDER BY kcu.table_name, kcu.column_name;


-- ════════════════════════════════════════════════════════════
-- SECTION 4: ENUM TYPES (allowed values for status columns etc)
-- ════════════════════════════════════════════════════════════
SELECT
    t.typname              AS "Enum Name",
    e.enumlabel            AS "Allowed Value",
    e.enumsortorder        AS "Order"
FROM pg_type t
JOIN pg_enum e ON t.oid = e.enumtypid
JOIN pg_catalog.pg_namespace n ON n.oid = t.typnamespace
WHERE n.nspname = 'public'
ORDER BY t.typname, e.enumsortorder;


-- ════════════════════════════════════════════════════════════
-- SECTION 5: INDEXES (fast lookup columns)
-- ════════════════════════════════════════════════════════════
SELECT
    tablename              AS "Table",
    indexname              AS "Index",
    indexdef               AS "Definition"
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;


-- ════════════════════════════════════════════════════════════
-- SECTION 6: ROW LEVEL SECURITY POLICIES
-- ════════════════════════════════════════════════════════════
SELECT
    tablename              AS "Table",
    policyname             AS "Policy Name",
    permissive             AS "Permissive",
    roles                  AS "Applies To Roles",
    cmd                    AS "Operation",
    qual                   AS "USING (row filter)",
    with_check             AS "WITH CHECK"
FROM pg_policies
WHERE schemaname = 'public'
ORDER BY tablename, policyname;


-- ════════════════════════════════════════════════════════════
-- SECTION 7: SAMPLE DATA — SELECT * (safe, no hardcoded cols)
-- Run each one separately if needed
-- ════════════════════════════════════════════════════════════

-- 7a. restaurants
SELECT * FROM restaurants LIMIT 3;

-- 7b. categories
SELECT * FROM categories LIMIT 5;

-- 7c. menu_items
SELECT * FROM menu_items LIMIT 5;

-- 7d. orders
SELECT * FROM orders LIMIT 5;

-- 7e. order_items  (actual column names discovered from Section 2)
SELECT * FROM order_items LIMIT 5;

-- 7f. restaurant_tables
SELECT * FROM restaurant_tables LIMIT 5;

-- 7g. profiles
SELECT * FROM profiles LIMIT 3;

-- 7h. promotions
SELECT * FROM promotions LIMIT 5;
