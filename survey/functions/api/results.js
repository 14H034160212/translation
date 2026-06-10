export async function onRequestGet({ request, env }) {
  const key = new URL(request.url).searchParams.get('key');

  if (!env.ADMIN_KEY || key !== env.ADMIN_KEY) {
    return new Response('Unauthorized', { status: 401 });
  }

  const { results } = await env.DB.prepare(
    `SELECT id, session_id, submitted_at, rater_name, jp_level, age_group, data
     FROM responses ORDER BY submitted_at`
  ).all();

  // One row per (rater, item, code) for easy analysis
  const rows = ['session_id,rater_name,jp_level,age_group,submitted_at,item,code,naturalness,similarity'];

  for (const row of results) {
    const data = JSON.parse(row.data);
    for (const [item, ratings] of Object.entries(data)) {
      for (const code of ['A', 'B', 'C', 'D']) {
        rows.push([
          row.session_id,
          `"${(row.rater_name || '').replace(/"/g, '""')}"`,
          row.jp_level,
          row.age_group,
          row.submitted_at,
          item,
          code,
          ratings[code] ?? '',
          code === 'A' ? (ratings.sim ?? '') : ''
        ].join(','));
      }
    }
  }

  return new Response(rows.join('\n'), {
    headers: {
      'Content-Type': 'text/csv',
      'Content-Disposition': 'attachment; filename="mos_results.csv"'
    }
  });
}
