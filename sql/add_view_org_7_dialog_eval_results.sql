CREATE ALGORITHM=UNDEFINED
DEFINER=`ss_mvp_dialog_analysis_admin`@`%`
SQL SECURITY DEFINER
VIEW `org_7_dialog_eval_results` AS
SELECT
  c.start_datetime           AS chunk_start_datetime,
  c.end_datetime             AS chunk_end_datetime,
  c.filename                 AS chunk_filename,
  c.chunk_id                 AS chunk_id,
  CAST(MAX(CASE WHEN ac.name = 'customers_count'            THEN CAST(v.value_text AS SIGNED) END) AS SIGNED) AS customers_count,
  CAST(MAX(CASE WHEN ac.name = 'standard_greeting'          THEN CAST(v.value_text AS SIGNED) END) AS SIGNED) AS standard_greeting,
  CAST(MAX(CASE WHEN ac.name = 'address_confirmed'          THEN CAST(v.value_text AS SIGNED) END) AS SIGNED) AS address_confirmed,
  CAST(MAX(CASE WHEN ac.name = 'name_asked_and_used'        THEN CAST(v.value_text AS SIGNED) END) AS SIGNED) AS name_asked_and_used,
  CAST(MAX(CASE WHEN ac.name = 'dialogue_navigation'        THEN CAST(v.value_text AS SIGNED) END) AS SIGNED) AS dialogue_navigation,
  CAST(MAX(CASE WHEN ac.name = 'helped_with_selection'      THEN CAST(v.value_text AS SIGNED) END) AS SIGNED) AS helped_with_selection,
  CAST(MAX(CASE WHEN ac.name = 'new_products_mentioned'      THEN CAST(v.value_text AS SIGNED) END) AS SIGNED) AS new_products_mentioned,
  CAST(MAX(CASE WHEN ac.name = 'promotions_mentioned'        THEN CAST(v.value_text AS SIGNED) END) AS SIGNED) AS promotions_mentioned,
  CAST(MAX(CASE WHEN ac.name = 'pizza_extras_offered'       THEN CAST(v.value_text AS SIGNED) END) AS SIGNED) AS pizza_extras_offered,
  CAST(MAX(CASE WHEN ac.name = 'additional_items_offered'   THEN CAST(v.value_text AS SIGNED) END) AS SIGNED) AS additional_items_offered,
  CAST(MAX(CASE WHEN ac.name = 'order_repeated'             THEN CAST(v.value_text AS SIGNED) END) AS SIGNED) AS order_repeated,
  CAST(MAX(CASE WHEN ac.name = 'amount_announced'           THEN CAST(v.value_text AS SIGNED) END) AS SIGNED) AS amount_announced,
  CAST(MAX(CASE WHEN ac.name = 'change_confirmed'           THEN CAST(v.value_text AS SIGNED) END) AS SIGNED) AS change_confirmed,
  CAST(MAX(CASE WHEN ac.name = 'delivery_time_announced'    THEN CAST(v.value_text AS SIGNED) END) AS SIGNED) AS delivery_time_announced,
  CAST(MAX(CASE WHEN ac.name = 'thanked_for_order'          THEN CAST(v.value_text AS SIGNED) END) AS SIGNED) AS thanked_for_order,
  CAST(MAX(CASE WHEN ac.name = 'farewell_instances'         THEN CAST(v.value_text AS SIGNED) END) AS SIGNED) AS farewell_instances,
  CAST(MAX(CASE WHEN ac.name = 'clear_pronunciation'        THEN CAST(v.value_text AS SIGNED) END) AS SIGNED) AS clear_pronunciation,
  CAST(MAX(CASE WHEN ac.name = 'politeness_instances'       THEN CAST(v.value_text AS SIGNED) END) AS SIGNED) AS politeness_instances,
  CAST(MAX(CASE WHEN ac.name = 'voice_tone'                 THEN CAST(v.value_text AS SIGNED) END) AS SIGNED) AS voice_tone,
  CAST(MAX(CASE WHEN ac.name = 'cash_payment'               THEN CAST(v.value_text AS SIGNED) END) AS SIGNED) AS cash_payment,
  c.organization_id           AS organization_id,
  c.sale_point_id             AS sale_point_id,
  sp.display_name             AS sale_point_display_name,
  av.created_at               AS audit_created_at,
  av.version_id               AS version_id
FROM dialog_chunks c
JOIN dialog_analysis da
  ON da.chunk_id = c.chunk_id
JOIN analysis_versions av
  ON av.version_id = da.version_id
    AND av.is_active = 1
-- выбираем только последнюю версию анализа для каждого чанка
JOIN (
  SELECT
    da.chunk_id    AS chunk_id,
    MAX(av.created_at) AS max_created
  FROM dialog_analysis da
  JOIN analysis_versions av
    ON av.version_id = da.version_id
      AND av.is_active = 1
  GROUP BY da.chunk_id
) last_av
  ON last_av.chunk_id = da.chunk_id
    AND last_av.max_created = av.created_at
LEFT JOIN dialog_analysis_values v
  ON v.analysis_id = da.id
LEFT JOIN audit_criteria ac
  ON ac.id = v.criterion_id
LEFT JOIN sale_points sp
  ON sp.sale_point_id = c.sale_point_id
WHERE c.organization_id = 7
GROUP BY c.chunk_id, av.version_id;
