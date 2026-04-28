-- Run THIS query in Supabase SQL Editor → get all columns of all tables
SELECT
    c.table_name        AS "Table",
    c.ordinal_position  AS "Pos",
    c.column_name       AS "Column",
    c.data_type         AS "Type",
    c.udt_name          AS "Subtype",
    c.is_nullable       AS "Nullable",
    c.column_default    AS "Default"
FROM information_schema.columns c
WHERE c.table_schema = 'public'
ORDER BY c.table_name, c.ordinal_position;
