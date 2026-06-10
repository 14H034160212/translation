CREATE TABLE IF NOT EXISTS responses (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id  TEXT    NOT NULL,
  submitted_at TEXT   DEFAULT (datetime('now')),
  rater_name  TEXT,
  jp_level    TEXT,
  age_group   TEXT,
  data        TEXT    NOT NULL  -- JSON: {item: {A:3, B:4, C:2, D:5, sim:"A"}, ...}
);
