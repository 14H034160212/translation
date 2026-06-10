// Blind code → system mapping (from manifest_rater1, same for all raters)
const CODE_MAP = {
  1: { A:'CosyVoice2', B:'F5-TTS',    C:'EdgeTTS',   D:'GPT-SoVITS' },
  2: { A:'EdgeTTS',   B:'CosyVoice2', C:'GPT-SoVITS', D:'F5-TTS'    },
  3: { A:'F5-TTS',    B:'EdgeTTS',    C:'CosyVoice2', D:'GPT-SoVITS' },
  4: { A:'F5-TTS',    B:'CosyVoice2', C:'GPT-SoVITS', D:'EdgeTTS'   },
  5: { A:'F5-TTS',    B:'CosyVoice2', C:'EdgeTTS',    D:'GPT-SoVITS' },
  6: { A:'F5-TTS',    B:'EdgeTTS',    C:'CosyVoice2', D:'GPT-SoVITS' },
  7: { A:'CosyVoice2', B:'F5-TTS',   C:'GPT-SoVITS', D:'EdgeTTS'   },
  8: { A:'F5-TTS',    B:'EdgeTTS',    C:'GPT-SoVITS', D:'CosyVoice2' },
};
const SYSTEMS = ['GPT-SoVITS', 'CosyVoice2', 'F5-TTS', 'EdgeTTS'];

export async function onRequestGet({ request, env }) {
  const key = new URL(request.url).searchParams.get('key');
  if (!env.ADMIN_KEY || key !== env.ADMIN_KEY) {
    return new Response(JSON.stringify({ error: 'Unauthorized' }), { status: 401 });
  }

  const { results } = await env.DB.prepare(
    `SELECT session_id, rater_name, jp_level, submitted_at, data FROM responses ORDER BY submitted_at`
  ).all();

  // ── Collect per-system naturalness scores and similarity votes ──
  const sysNat  = Object.fromEntries(SYSTEMS.map(s => [s, []]));
  const sysSim  = Object.fromEntries(SYSTEMS.map(s => [s, 0]));
  const raters  = [];

  // For ICC: matrix[subject_idx][rater_idx] = naturalness score
  // subject = (item, system), rater = session_id order
  const raterIndex = {};
  const subjectScores = {}; // key: "item_system" → [scores per rater]

  for (const row of results) {
    const data = JSON.parse(row.data);
    const rid = row.session_id;
    if (!raterIndex[rid]) {
      raterIndex[rid] = Object.keys(raterIndex).length;
      raters.push({ id: rid, name: row.rater_name, jp_level: row.jp_level, submitted_at: row.submitted_at });
    }
    const ri = raterIndex[rid];

    for (const [itemStr, ratings] of Object.entries(data)) {
      const item = parseInt(itemStr);
      const map  = CODE_MAP[item] || {};
      for (const code of ['A', 'B', 'C', 'D']) {
        const sys = map[code];
        if (!sys) continue;
        const score = ratings[code];
        if (score) {
          sysNat[sys].push(score);
          const key = `${item}_${sys}`;
          if (!subjectScores[key]) subjectScores[key] = [];
          subjectScores[key][ri] = score;
        }
      }
      // Similarity: which system got the most votes
      if (ratings.sim && ratings.sim !== 'none') {
        const sys = map[ratings.sim];
        if (sys) sysSim[sys]++;
      }
    }
  }

  // ── Stats: mean, sd ──
  const stats = {};
  for (const sys of SYSTEMS) {
    const vals = sysNat[sys];
    const n    = vals.length;
    const mean = n ? vals.reduce((a,b) => a+b, 0) / n : null;
    const sd   = n > 1 ? Math.sqrt(vals.reduce((s,v) => s+(v-mean)**2, 0) / (n-1)) : 0;
    stats[sys] = { mean: mean ? +mean.toFixed(3) : null, sd: +sd.toFixed(3), n, sim: sysSim[sys] };
  }

  // ── ICC(2,1) two-way random, single measures ──
  const nRaters = Object.keys(raterIndex).length;
  let icc = null;
  if (nRaters >= 2) {
    const subjects = Object.keys(subjectScores)
      .filter(k => subjectScores[k].filter(v => v !== undefined).length === nRaters);
    const nSubj = subjects.length;
    if (nSubj >= 2) {
      const mat = subjects.map(k => {
        const row = [];
        for (let r = 0; r < nRaters; r++) row.push(subjectScores[k][r] ?? 0);
        return row;
      });
      const grandMean = mat.flat().reduce((a,b)=>a+b,0) / (nSubj * nRaters);
      const ssRows = nRaters * mat.reduce((s,r)=> s + (r.reduce((a,b)=>a+b,0)/nRaters - grandMean)**2, 0);
      const ssCols = nSubj  * Array.from({length:nRaters},(_,j)=>mat.reduce((s,r)=>s+r[j],0)/nSubj)
                              .reduce((s,c)=> s+(c-grandMean)**2, 0);
      const ssTotal = mat.flat().reduce((s,v)=> s+(v-grandMean)**2, 0);
      const ssErr   = ssTotal - ssRows - ssCols;
      const msRows  = ssRows / (nSubj - 1);
      const msCols  = ssCols / (nRaters - 1);
      const msErr   = ssErr  / ((nSubj-1)*(nRaters-1));
      icc = (msRows - msErr) / (msRows + (nRaters-1)*msErr + nRaters*(msCols-msErr)/nSubj);
      icc = +icc.toFixed(3);
    }
  }

  return new Response(JSON.stringify({ stats, icc, raters, n_responses: results.length }), {
    headers: { 'Content-Type': 'application/json' }
  });
}
