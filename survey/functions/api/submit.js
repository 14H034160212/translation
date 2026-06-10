export async function onRequestPost({ request, env }) {
  try {
    const { session_id, rater_name, jp_level, age_group, data } = await request.json();

    if (!session_id || !data) {
      return json({ error: 'Missing required fields' }, 400);
    }

    await env.DB.prepare(
      `INSERT INTO responses (session_id, rater_name, jp_level, age_group, data)
       VALUES (?, ?, ?, ?, ?)`
    ).bind(session_id, rater_name || '', jp_level || '', age_group || '', JSON.stringify(data)).run();

    return json({ ok: true }, 200);
  } catch (e) {
    return json({ error: e.message }, 500);
  }
}

function json(obj, status) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { 'Content-Type': 'application/json' }
  });
}
