"""SQLite storage for the Smart Election demo.

Integrity rules are enforced by the database itself (triggers), not just the UI:
  * Ballots, turnout records and the audit log can never be updated or deleted.
  * Ballots can only be added while their election is open.
  * An election only moves forward: setup -> open -> closed (never reopened).
  * Candidates can only be changed while the election is in setup.
  * A voter who has voted in any election can never be deleted.
Ballots and audit entries are hash-chained, so editing the file by hand is detectable.

Secret ballot: ballots store only the candidate. The turnout table (who voted)
is WITHOUT ROWID, so its storage order can't be used to link voters to ballots.
"""

import hashlib
import json
import secrets
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DB_PATH = Path("data") / "voting_demo.db"
DB_PATH.parent.mkdir(exist_ok=True)

NOTA = "NOTA"
DEFAULT_CANDIDATES = ["BJP", "Congress", "JDS"]
GENESIS_HASH = "0" * 64
MAX_NAME_LENGTH = 40

SCHEMA = """
CREATE TABLE IF NOT EXISTS voters (
    voter_id      TEXT PRIMARY KEY,
    registered_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS elections (
    election_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT NOT NULL,
    state          TEXT NOT NULL DEFAULT 'setup' CHECK (state IN ('setup', 'open', 'closed')),
    created_at     TEXT NOT NULL,
    opened_at      TEXT,
    closed_at      TEXT,
    eligible_count INTEGER
);

CREATE TABLE IF NOT EXISTS candidates (
    candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
    election_id  INTEGER NOT NULL REFERENCES elections(election_id),
    name         TEXT NOT NULL,
    UNIQUE (election_id, name)
);

CREATE TABLE IF NOT EXISTS turnout (
    election_id INTEGER NOT NULL REFERENCES elections(election_id),
    voter_id    TEXT NOT NULL REFERENCES voters(voter_id),
    PRIMARY KEY (election_id, voter_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS ballots (
    ballot_id   INTEGER PRIMARY KEY,
    election_id INTEGER NOT NULL REFERENCES elections(election_id),
    candidate   TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    prev_hash   TEXT NOT NULL,
    hash        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    log_id     INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    actor      TEXT NOT NULL,
    event      TEXT NOT NULL,
    details    TEXT NOT NULL DEFAULT '',
    prev_hash  TEXT NOT NULL,
    hash       TEXT NOT NULL
);

-- Ballots: permanent, only while open, only for real candidates.
CREATE TRIGGER IF NOT EXISTS ballots_no_update BEFORE UPDATE ON ballots
BEGIN SELECT RAISE(ABORT, 'Ballots are permanent and cannot be changed.'); END;
CREATE TRIGGER IF NOT EXISTS ballots_no_delete BEFORE DELETE ON ballots
BEGIN SELECT RAISE(ABORT, 'Ballots are permanent and cannot be deleted.'); END;
CREATE TRIGGER IF NOT EXISTS ballots_only_when_open BEFORE INSERT ON ballots
WHEN (SELECT state FROM elections WHERE election_id = NEW.election_id) IS NOT 'open'
BEGIN SELECT RAISE(ABORT, 'Voting is not open.'); END;
CREATE TRIGGER IF NOT EXISTS ballots_valid_candidate BEFORE INSERT ON ballots
WHEN NOT EXISTS (SELECT 1 FROM candidates WHERE election_id = NEW.election_id AND name = NEW.candidate)
BEGIN SELECT RAISE(ABORT, 'Invalid candidate.'); END;

-- Turnout ("who voted"): permanent, only while open.
CREATE TRIGGER IF NOT EXISTS turnout_no_update BEFORE UPDATE ON turnout
BEGIN SELECT RAISE(ABORT, 'Turnout records are permanent.'); END;
CREATE TRIGGER IF NOT EXISTS turnout_no_delete BEFORE DELETE ON turnout
BEGIN SELECT RAISE(ABORT, 'Turnout records are permanent.'); END;
CREATE TRIGGER IF NOT EXISTS turnout_only_when_open BEFORE INSERT ON turnout
WHEN (SELECT state FROM elections WHERE election_id = NEW.election_id) IS NOT 'open'
BEGIN SELECT RAISE(ABORT, 'Voting is not open.'); END;

-- Audit log: append-only.
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'The audit log is append-only.'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'The audit log is append-only.'); END;

-- Elections: never deleted, only move forward, frozen once closed.
CREATE TRIGGER IF NOT EXISTS elections_no_delete BEFORE DELETE ON elections
BEGIN SELECT RAISE(ABORT, 'Elections cannot be deleted.'); END;
CREATE TRIGGER IF NOT EXISTS elections_closed_frozen BEFORE UPDATE ON elections
WHEN OLD.state = 'closed'
BEGIN SELECT RAISE(ABORT, 'A closed election cannot be changed or reopened.'); END;
CREATE TRIGGER IF NOT EXISTS elections_forward_only BEFORE UPDATE ON elections
WHEN NOT ((OLD.state = 'setup' AND NEW.state IN ('setup', 'open'))
       OR (OLD.state = 'open'  AND NEW.state IN ('open', 'closed')))
BEGIN SELECT RAISE(ABORT, 'Invalid election state change.'); END;
CREATE TRIGGER IF NOT EXISTS elections_name_locked BEFORE UPDATE OF name ON elections
WHEN OLD.state <> 'setup' AND NEW.name <> OLD.name
BEGIN SELECT RAISE(ABORT, 'The election name is locked once voting opens.'); END;

-- Candidates: editable only during setup.
CREATE TRIGGER IF NOT EXISTS candidates_locked_insert BEFORE INSERT ON candidates
WHEN (SELECT state FROM elections WHERE election_id = NEW.election_id) IS NOT 'setup'
BEGIN SELECT RAISE(ABORT, 'Candidates are locked once voting opens.'); END;
CREATE TRIGGER IF NOT EXISTS candidates_locked_update BEFORE UPDATE ON candidates
WHEN (SELECT state FROM elections WHERE election_id = OLD.election_id) IS NOT 'setup'
BEGIN SELECT RAISE(ABORT, 'Candidates are locked once voting opens.'); END;
CREATE TRIGGER IF NOT EXISTS candidates_locked_delete BEFORE DELETE ON candidates
WHEN (SELECT state FROM elections WHERE election_id = OLD.election_id) IS NOT 'setup'
BEGIN SELECT RAISE(ABORT, 'Candidates are locked once voting opens.'); END;

-- Voters who have voted can never be removed.
CREATE TRIGGER IF NOT EXISTS voters_keep_if_voted BEFORE DELETE ON voters
WHEN EXISTS (SELECT 1 FROM turnout WHERE voter_id = OLD.voter_id)
BEGIN SELECT RAISE(ABORT, 'This voter has voted and cannot be removed.'); END;

-- Self-registration applications. ID numbers are never stored in full:
-- only a masked copy (for display) and a keyed hash (to detect duplicates).
CREATE TABLE IF NOT EXISTS registrations (
    ref           TEXT PRIMARY KEY,
    full_name     TEXT NOT NULL,
    dob           TEXT NOT NULL,
    gender        TEXT NOT NULL,
    id_type       TEXT NOT NULL CHECK (id_type IN ('epic', 'aadhaar')),
    id_masked     TEXT NOT NULL,
    id_hash       TEXT NOT NULL,
    mobile        TEXT,
    status        TEXT NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending', 'approved', 'rejected', 'removed')),
    checks        TEXT NOT NULL,
    flags         INTEGER NOT NULL DEFAULT 0,
    duplicate_of  TEXT,
    submitted_at  TEXT NOT NULL,
    reviewed_at   TEXT,
    reviewed_by   TEXT,
    reject_reason TEXT
);
CREATE INDEX IF NOT EXISTS registrations_id_hash ON registrations (id_hash);

-- Vote receipts: lets a voter check their vote was counted WITHOUT revealing the choice.
-- Stores only a hash of the voter's secret code and the election. WITHOUT ROWID + random
-- primary key means storage order can't be matched to ballot order.
CREATE TABLE IF NOT EXISTS receipts (
    receipt_hash TEXT PRIMARY KEY,
    election_id  INTEGER NOT NULL REFERENCES elections(election_id)
) WITHOUT ROWID;
CREATE TRIGGER IF NOT EXISTS receipts_no_update BEFORE UPDATE ON receipts
BEGIN SELECT RAISE(ABORT, 'Receipts are permanent.'); END;
CREATE TRIGGER IF NOT EXISTS receipts_no_delete BEFORE DELETE ON receipts
BEGIN SELECT RAISE(ABORT, 'Receipts are permanent.'); END;

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS registrations_no_delete BEFORE DELETE ON registrations
BEGIN SELECT RAISE(ABORT, 'Registration records are kept permanently.'); END;
CREATE TRIGGER IF NOT EXISTS registrations_forward_only BEFORE UPDATE OF status ON registrations
WHEN NOT (OLD.status = NEW.status
       OR (OLD.status = 'pending'  AND NEW.status IN ('approved', 'rejected'))
       OR (OLD.status = 'approved' AND NEW.status = 'removed'))
BEGIN SELECT RAISE(ABORT, 'A decided registration cannot be changed.'); END;
"""


class ElectionError(ValueError):
    """A rule of the election was broken (shown to the user as-is)."""

    def __init__(self, message, code="error"):
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------
# Connections
# --------------------------------------------------------------------------

def _connect():
    connection = sqlite3.connect(DB_PATH, timeout=10, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def get_connection():
    """Read-only use: open a connection and always close it."""
    connection = _connect()
    try:
        yield connection
    finally:
        connection.close()


@contextmanager
def transaction():
    """Write use: BEGIN IMMEDIATE ... COMMIT, rolled back on any error."""
    connection = _connect()
    try:
        connection.execute("BEGIN IMMEDIATE")
        yield connection
        connection.execute("COMMIT")
    except sqlite3.IntegrityError as error:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise ElectionError(str(error), "integrity") from error
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _hash(*parts):
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()


# --------------------------------------------------------------------------
# Hash-chained append-only tables
# --------------------------------------------------------------------------

def log_event(connection, actor, event, details=""):
    """Append an audit entry. Must be called inside transaction()."""
    last = connection.execute(
        "SELECT log_id, hash FROM audit_log ORDER BY log_id DESC LIMIT 1"
    ).fetchone()
    log_id = (last["log_id"] + 1) if last else 1
    prev_hash = last["hash"] if last else GENESIS_HASH
    created_at = now()
    digest = _hash(prev_hash, log_id, created_at, actor, event, details)
    connection.execute(
        "INSERT INTO audit_log (log_id, created_at, actor, event, details, prev_hash, hash)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (log_id, created_at, actor, event, details, prev_hash, digest),
    )


def _append_ballot(connection, election_id, candidate, created_at=None):
    last = connection.execute(
        "SELECT ballot_id, hash FROM ballots ORDER BY ballot_id DESC LIMIT 1"
    ).fetchone()
    ballot_id = (last["ballot_id"] + 1) if last else 1
    prev_hash = last["hash"] if last else GENESIS_HASH
    created_at = created_at or now()
    digest = _hash(prev_hash, ballot_id, election_id, candidate, created_at)
    connection.execute(
        "INSERT INTO ballots (ballot_id, election_id, candidate, created_at, prev_hash, hash)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (ballot_id, election_id, candidate, created_at, prev_hash, digest),
    )


def verify_ballot_chain():
    """Return (ok, message, count). Detects any edited, removed or inserted ballot."""
    with get_connection() as connection:
        rows = connection.execute("SELECT * FROM ballots ORDER BY ballot_id").fetchall()
    prev_hash = GENESIS_HASH
    for expected_id, row in enumerate(rows, start=1):
        if row["ballot_id"] != expected_id:
            return False, f"Ballot #{expected_id} is missing.", len(rows)
        digest = _hash(prev_hash, row["ballot_id"], row["election_id"], row["candidate"], row["created_at"])
        if row["prev_hash"] != prev_hash or row["hash"] != digest:
            return False, f"Ballot #{row['ballot_id']} has been altered.", len(rows)
        prev_hash = row["hash"]
    return True, f"All {len(rows)} ballots (across all elections) verified.", len(rows)


def verify_audit_chain():
    with get_connection() as connection:
        rows = connection.execute("SELECT * FROM audit_log ORDER BY log_id").fetchall()
    prev_hash = GENESIS_HASH
    for expected_id, row in enumerate(rows, start=1):
        if row["log_id"] != expected_id:
            return False, f"Audit entry #{expected_id} is missing.", len(rows)
        digest = _hash(prev_hash, row["log_id"], row["created_at"], row["actor"], row["event"], row["details"])
        if row["prev_hash"] != prev_hash or row["hash"] != digest:
            return False, f"Audit entry #{row['log_id']} has been altered.", len(rows)
        prev_hash = row["hash"]
    return True, f"All {len(rows)} audit entries verified.", len(rows)


# --------------------------------------------------------------------------
# Setup and migration from the old two-table schema
# --------------------------------------------------------------------------

def _tables(connection):
    return {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def _columns(connection, table):
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def initialize_database():
    connection = _connect()
    try:
        tables = _tables(connection)
        old_schema = "participation" in tables or (
            "ballots" in tables and "election_id" not in _columns(connection, "ballots")
        )
        if old_schema:
            backup = DB_PATH.with_name(DB_PATH.stem + ".pre-migration.db")
            if not backup.exists():
                connection.close()
                shutil.copy2(DB_PATH, backup)
                connection = _connect()
            connection.execute("BEGIN IMMEDIATE")
            if "participation" in tables:
                connection.execute("ALTER TABLE participation RENAME TO legacy_participation")
            if "ballots" in tables:
                connection.execute("ALTER TABLE ballots RENAME TO legacy_ballots")
            connection.execute("COMMIT")

        connection.executescript(SCHEMA)
        voter_columns = _columns(connection, "voters")
        if "full_name" not in voter_columns:
            connection.execute("ALTER TABLE voters ADD COLUMN full_name TEXT")
        if "source" not in voter_columns:
            connection.execute("ALTER TABLE voters ADD COLUMN source TEXT NOT NULL DEFAULT 'officer'")
    finally:
        connection.close()

    with transaction() as connection:
        if connection.execute("SELECT COUNT(*) FROM elections").fetchone()[0]:
            return
        tables = _tables(connection)
        if "legacy_participation" in tables or "legacy_ballots" in tables:
            _migrate_legacy(connection, tables)
        else:
            _create_election(connection, "Election 1", DEFAULT_CANDIDATES, "system")


def _create_election(connection, name, candidate_names, actor):
    cursor = connection.execute(
        "INSERT INTO elections (name, created_at) VALUES (?, ?)", (name, now())
    )
    election_id = cursor.lastrowid
    for candidate in [c for c in candidate_names if c != NOTA] + [NOTA]:
        connection.execute(
            "INSERT INTO candidates (election_id, name) VALUES (?, ?)", (election_id, candidate)
        )
    log_event(connection, actor, "Election created", f"{name} (#{election_id})")
    return election_id


def _migrate_legacy(connection, tables):
    legacy_ballots = []
    if "legacy_ballots" in tables:
        legacy_ballots = connection.execute(
            "SELECT candidate, created_at FROM legacy_ballots ORDER BY ballot_id"
        ).fetchall()

    names = list(DEFAULT_CANDIDATES)
    for row in legacy_ballots:
        if row["candidate"] not in names and row["candidate"] != NOTA:
            names.append(row["candidate"])
    election_id = _create_election(connection, "Election 1", names, "system")

    voted = []
    if "legacy_participation" in tables:
        for row in connection.execute("SELECT voter_id, has_voted FROM legacy_participation"):
            connection.execute(
                "INSERT OR IGNORE INTO voters (voter_id, registered_at) VALUES (?, ?)",
                (row["voter_id"], now()),
            )
            if row["has_voted"]:
                voted.append(row["voter_id"])

    if legacy_ballots or voted:
        opened = legacy_ballots[0]["created_at"] if legacy_ballots else now()
        connection.execute(
            "UPDATE elections SET state = 'open', opened_at = ? WHERE election_id = ?",
            (opened, election_id),
        )
        for voter_id in voted:
            connection.execute(
                "INSERT INTO turnout (election_id, voter_id) VALUES (?, ?)", (election_id, voter_id)
            )
        for row in legacy_ballots:
            _append_ballot(connection, election_id, row["candidate"], row["created_at"])

    log_event(
        connection, "system", "Database upgraded",
        f"Imported {len(voted)} turnout records and {len(legacy_ballots)} ballots into Election 1",
    )


# --------------------------------------------------------------------------
# Elections and candidates
# --------------------------------------------------------------------------

def _election_dict(row):
    return dict(row) if row else None


def current_election(connection=None):
    """The latest election (there is only ever one that isn't closed)."""
    query = "SELECT * FROM elections ORDER BY election_id DESC LIMIT 1"
    if connection is not None:
        return _election_dict(connection.execute(query).fetchone())
    with get_connection() as conn:
        return _election_dict(conn.execute(query).fetchone())


def get_election(election_id):
    with get_connection() as connection:
        return _election_dict(connection.execute(
            "SELECT * FROM elections WHERE election_id = ?", (election_id,)
        ).fetchone())


def list_elections():
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT e.*, (SELECT COUNT(*) FROM ballots b WHERE b.election_id = e.election_id) AS ballots
            FROM elections e ORDER BY e.election_id DESC
        """).fetchall()
    return [dict(row) for row in rows]


def get_candidates(election_id=None, connection=None):
    def run(conn):
        eid = election_id or current_election(conn)["election_id"]
        rows = conn.execute(
            "SELECT name FROM candidates WHERE election_id = ? ORDER BY (name = ?), candidate_id",
            (eid, NOTA),
        ).fetchall()
        return [row["name"] for row in rows]

    if connection is not None:
        return run(connection)
    with get_connection() as conn:
        return run(conn)


def _clean_name(name, what="Name"):
    name = " ".join(str(name or "").split())
    if not name:
        raise ElectionError(f"{what} cannot be empty.")
    if len(name) > MAX_NAME_LENGTH:
        raise ElectionError(f"{what} must be at most {MAX_NAME_LENGTH} characters.")
    return name


def _require_setup(connection):
    election = current_election(connection)
    if election["state"] != "setup":
        raise ElectionError("Candidates are locked once voting has opened.", "locked")
    return election


def add_candidate(name, actor):
    name = _clean_name(name, "Candidate name")
    with transaction() as connection:
        election = _require_setup(connection)
        if name.upper() == NOTA or name in get_candidates(election["election_id"], connection):
            raise ElectionError(f"'{name}' is already on the ballot.")
        connection.execute(
            "INSERT INTO candidates (election_id, name) VALUES (?, ?)", (election["election_id"], name)
        )
        log_event(connection, actor, "Candidate added", name)


def rename_candidate(old_name, new_name, actor):
    new_name = _clean_name(new_name, "Candidate name")
    with transaction() as connection:
        election = _require_setup(connection)
        names = get_candidates(election["election_id"], connection)
        if old_name == NOTA:
            raise ElectionError("NOTA cannot be renamed.")
        if old_name not in names:
            raise ElectionError("Candidate not found.")
        if new_name == old_name:
            return
        if new_name.upper() == NOTA or new_name in names:
            raise ElectionError(f"'{new_name}' is already on the ballot.")
        connection.execute(
            "UPDATE candidates SET name = ? WHERE election_id = ? AND name = ?",
            (new_name, election["election_id"], old_name),
        )
        log_event(connection, actor, "Candidate renamed", f"{old_name} -> {new_name}")


def remove_candidate(name, actor):
    with transaction() as connection:
        election = _require_setup(connection)
        if name == NOTA:
            raise ElectionError("NOTA is always on the ballot.")
        cursor = connection.execute(
            "DELETE FROM candidates WHERE election_id = ? AND name = ?", (election["election_id"], name)
        )
        if cursor.rowcount == 0:
            raise ElectionError("Candidate not found.")
        log_event(connection, actor, "Candidate removed", name)


def rename_election(name, actor):
    name = _clean_name(name, "Election name")
    with transaction() as connection:
        election = current_election(connection)
        if election["state"] != "setup":
            raise ElectionError("The election name is locked once voting has opened.", "locked")
        connection.execute("UPDATE elections SET name = ? WHERE election_id = ?", (name, election["election_id"]))
        log_event(connection, actor, "Election renamed", f"{election['name']} -> {name}")


def open_election(actor):
    with transaction() as connection:
        election = current_election(connection)
        if election["state"] != "setup":
            raise ElectionError("Voting can only be opened from setup.")
        names = get_candidates(election["election_id"], connection)
        if len([n for n in names if n != NOTA]) < 1:
            raise ElectionError("Add at least one candidate before opening voting.")
        connection.execute(
            "UPDATE elections SET state = 'open', opened_at = ? WHERE election_id = ?",
            (now(), election["election_id"]),
        )
        log_event(connection, actor, "Voting opened", f"{election['name']} - candidates: {', '.join(names)}")


def close_election(actor):
    with transaction() as connection:
        election = current_election(connection)
        if election["state"] != "open":
            raise ElectionError("Only an open election can be closed.")
        eligible = connection.execute("SELECT COUNT(*) FROM voters").fetchone()[0]
        connection.execute(
            "UPDATE elections SET state = 'closed', closed_at = ?, eligible_count = ? WHERE election_id = ?",
            (now(), eligible, election["election_id"]),
        )
        ballots = connection.execute(
            "SELECT COUNT(*) FROM ballots WHERE election_id = ?", (election["election_id"],)
        ).fetchone()[0]
        log_event(connection, actor, "Voting closed", f"{election['name']} - {ballots} ballots")


def new_election(name, actor):
    name = _clean_name(name, "Election name")
    with transaction() as connection:
        election = current_election(connection)
        if election["state"] != "closed":
            raise ElectionError("Close the current election before starting a new one.")
        names = get_candidates(election["election_id"], connection)
        return _create_election(connection, name, names, actor)


# --------------------------------------------------------------------------
# Voters
# --------------------------------------------------------------------------

def register_voter(voter_id, actor="admin", full_name=None):
    with transaction() as connection:
        cursor = connection.execute(
            "INSERT OR IGNORE INTO voters (voter_id, registered_at, full_name) VALUES (?, ?, ?)",
            (voter_id, now(), full_name),
        )
        if cursor.rowcount:
            log_event(connection, actor, "Voter registered", voter_id)


def log_face_update(voter_id, actor):
    with transaction() as connection:
        log_event(connection, actor, "Voter face re-captured", voter_id)


def delete_voter(voter_id, actor):
    with transaction() as connection:
        cursor = connection.execute("DELETE FROM voters WHERE voter_id = ?", (voter_id,))
        if cursor.rowcount == 0:
            raise ElectionError("Voter not found.")
        # Free their ID number so they could apply again.
        connection.execute(
            "UPDATE registrations SET status = 'removed' WHERE ref = ? AND status = 'approved'", (voter_id,)
        )
        log_event(connection, actor, "Voter removed", voter_id)


def voter_name(voter_id):
    with get_connection() as connection:
        row = connection.execute("SELECT full_name FROM voters WHERE voter_id = ?", (voter_id,)).fetchone()
    return row["full_name"] if row else None


def is_registered(voter_id):
    with get_connection() as connection:
        return connection.execute(
            "SELECT 1 FROM voters WHERE voter_id = ?", (voter_id,)
        ).fetchone() is not None


def has_voted(voter_id, election_id=None):
    with get_connection() as connection:
        eid = election_id or current_election(connection)["election_id"]
        return connection.execute(
            "SELECT 1 FROM turnout WHERE election_id = ? AND voter_id = ?", (eid, voter_id)
        ).fetchone() is not None


def list_voters():
    """Voters with current-election status and whether they ever voted."""
    with get_connection() as connection:
        eid = current_election(connection)["election_id"]
        rows = connection.execute("""
            SELECT v.voter_id, v.registered_at, v.full_name, v.source,
                   r.id_type, r.id_masked,
                   EXISTS (SELECT 1 FROM turnout t WHERE t.voter_id = v.voter_id AND t.election_id = ?) AS voted_now,
                   EXISTS (SELECT 1 FROM turnout t WHERE t.voter_id = v.voter_id) AS ever_voted
            FROM voters v LEFT JOIN registrations r ON r.ref = v.voter_id
            ORDER BY v.voter_id COLLATE NOCASE
        """, (eid,)).fetchall()
    return [dict(row) for row in rows]


def voter_count():
    with get_connection() as connection:
        return connection.execute("SELECT COUNT(*) FROM voters").fetchone()[0]


# --------------------------------------------------------------------------
# Self-registration applications
# --------------------------------------------------------------------------

def _next_ref(connection):
    number = connection.execute("SELECT COUNT(*) FROM registrations").fetchone()[0] + 1
    while True:
        ref = f"SE-{number:04d}"
        taken = connection.execute(
            "SELECT 1 FROM registrations WHERE ref = ? UNION SELECT 1 FROM voters WHERE voter_id = ?", (ref, ref)
        ).fetchone()
        if not taken:
            return ref
        number += 1


def id_in_use(id_hash):
    """Reference of a pending/approved application using this ID number, if any."""
    with get_connection() as connection:
        row = connection.execute(
            "SELECT ref FROM registrations WHERE id_hash = ? AND status IN ('pending', 'approved')", (id_hash,)
        ).fetchone()
    return row["ref"] if row else None


AUTO_APPROVER = "auto-check"


def get_setting(key, default=None):
    with get_connection() as connection:
        row = connection.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(key, value, actor):
    with transaction() as connection:
        connection.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        log_event(connection, actor, "Setting changed", f"{key} = {value}")


def auto_approve_enabled():
    return get_setting("auto_approve", "on") == "on"


def create_application(details, checks, flags, duplicate_of, ref_callback, auto_approve_face=None):
    """Insert an application. ref_callback(ref) saves the face/ID files inside the transaction,
    so a failure there leaves no half-created application behind.

    If auto_approve_face is given (all checks passed and auto-approval is on), the voter is
    approved in the same transaction and auto_approve_face(ref) moves their face into the
    voting dataset. Returns (ref, status)."""
    with transaction() as connection:
        if connection.execute(
            "SELECT 1 FROM registrations WHERE id_hash = ? AND status IN ('pending', 'approved')",
            (details["id_hash"],),
        ).fetchone():
            raise ElectionError("This ID number is already registered.", "duplicate_id")
        ref = _next_ref(connection)
        ref_callback(ref)
        connection.execute("""
            INSERT INTO registrations (ref, full_name, dob, gender, id_type, id_masked, id_hash, mobile,
                                       checks, flags, duplicate_of, submitted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ref, details["full_name"], details["dob"], details["gender"], details["id_type"],
              details["id_masked"], details["id_hash"], details.get("mobile") or None,
              json.dumps(checks), flags, duplicate_of, now()))
        log_event(connection, "public", "Registration submitted",
                  f"{ref} - {flags} flag{'' if flags == 1 else 's'}")
        if auto_approve_face is None:
            return ref, "pending"
        connection.execute(
            "UPDATE registrations SET status = 'approved', reviewed_at = ?, reviewed_by = ? WHERE ref = ?",
            (now(), AUTO_APPROVER, ref),
        )
        connection.execute(
            "INSERT INTO voters (voter_id, registered_at, full_name, source) VALUES (?, ?, ?, 'self')",
            (ref, now(), details["full_name"]),
        )
        log_event(connection, AUTO_APPROVER, "Registration auto-approved", f"{ref} - all checks passed")
        auto_approve_face(ref)
        return ref, "approved"


def _application_dict(row):
    if not row:
        return None
    data = dict(row)
    data["checks"] = json.loads(data["checks"])
    return data


def get_application(ref):
    with get_connection() as connection:
        return _application_dict(connection.execute(
            "SELECT * FROM registrations WHERE ref = ?", (ref,)
        ).fetchone())


def list_applications(status=None):
    with get_connection() as connection:
        if status == "auto":
            rows = connection.execute(
                "SELECT * FROM registrations WHERE reviewed_by = ? ORDER BY submitted_at DESC", (AUTO_APPROVER,)
            ).fetchall()
        elif status:
            rows = connection.execute(
                "SELECT * FROM registrations WHERE status = ? ORDER BY submitted_at DESC", (status,)
            ).fetchall()
        else:
            rows = connection.execute("SELECT * FROM registrations ORDER BY submitted_at DESC").fetchall()
    return [_application_dict(row) for row in rows]


def application_counts():
    with get_connection() as connection:
        rows = connection.execute("SELECT status, COUNT(*) FROM registrations GROUP BY status").fetchall()
        auto = connection.execute(
            "SELECT COUNT(*) FROM registrations WHERE reviewed_by = ?", (AUTO_APPROVER,)
        ).fetchone()[0]
    counts = {"pending": 0, "approved": 0, "rejected": 0, "removed": 0}
    counts.update({row[0]: row[1] for row in rows})
    counts["auto"] = auto
    return counts


def approve_application(ref, actor, add_face):
    """Approve: add_face(ref) moves the applicant's face into the voter dataset."""
    with transaction() as connection:
        app = connection.execute("SELECT * FROM registrations WHERE ref = ?", (ref,)).fetchone()
        if not app:
            raise ElectionError("Application not found.")
        if app["status"] != "pending":
            raise ElectionError("This application has already been decided.")
        connection.execute(
            "UPDATE registrations SET status = 'approved', reviewed_at = ?, reviewed_by = ? WHERE ref = ?",
            (now(), actor, ref),
        )
        connection.execute(
            "INSERT INTO voters (voter_id, registered_at, full_name, source) VALUES (?, ?, ?, 'self')",
            (ref, now(), app["full_name"]),
        )
        log_event(connection, actor, "Registration approved", ref)
        add_face(ref)


def reject_application(ref, reason, actor):
    reason = " ".join(str(reason or "").split())[:200]
    if not reason:
        raise ElectionError("Give a reason for rejecting.")
    with transaction() as connection:
        app = connection.execute("SELECT status FROM registrations WHERE ref = ?", (ref,)).fetchone()
        if not app:
            raise ElectionError("Application not found.")
        if app["status"] != "pending":
            raise ElectionError("This application has already been decided.")
        connection.execute(
            "UPDATE registrations SET status = 'rejected', reviewed_at = ?, reviewed_by = ?, reject_reason = ?"
            " WHERE ref = ?", (now(), actor, reason, ref),
        )
        log_event(connection, actor, "Registration rejected", f"{ref} - {reason}")


# --------------------------------------------------------------------------
# Voting and results
# --------------------------------------------------------------------------

def record_ballot(voter_id, candidate):
    """Record one secret ballot and mark the voter as having voted — atomically."""
    with transaction() as connection:
        election = current_election(connection)
        if election["state"] != "open":
            raise ElectionError("Voting is not open.", "not_open")
        if candidate not in get_candidates(election["election_id"], connection):
            raise ElectionError("Invalid candidate.", "invalid_candidate")
        if connection.execute("SELECT 1 FROM voters WHERE voter_id = ?", (voter_id,)).fetchone() is None:
            raise ElectionError("Voter is not registered.", "not_registered")
        if connection.execute(
            "SELECT 1 FROM turnout WHERE election_id = ? AND voter_id = ?",
            (election["election_id"], voter_id),
        ).fetchone():
            raise ElectionError("This voter has already voted.", "already_voted")

        connection.execute(
            "INSERT INTO turnout (election_id, voter_id) VALUES (?, ?)", (election["election_id"], voter_id)
        )
        _append_ballot(connection, election["election_id"], candidate)
        receipt = _new_receipt(connection, election["election_id"])
        # No voter ID here: logging who voted *when* could reveal how they voted.
        log_event(connection, "booth", "Ballot cast", election["name"])
        return receipt


RECEIPT_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"   # no 0/O or 1/I/L, easy to read aloud


def _receipt_hash(code):
    return hashlib.sha256(("receipt|" + code).encode()).hexdigest()


def normalise_receipt(code):
    return "".join(ch for ch in str(code or "").upper() if ch in RECEIPT_ALPHABET)


def _new_receipt(connection, election_id):
    code = "".join(secrets.choice(RECEIPT_ALPHABET) for _ in range(12))
    connection.execute("INSERT INTO receipts (receipt_hash, election_id) VALUES (?, ?)",
                       (_receipt_hash(code), election_id))
    return "-".join(code[i:i + 4] for i in range(0, 12, 4))


def verify_receipt(code):
    """The election a receipt belongs to, or None. Never reveals the vote itself."""
    code = normalise_receipt(code)
    if len(code) != 12:
        return None
    with get_connection() as connection:
        row = connection.execute("""
            SELECT e.* FROM receipts r JOIN elections e ON e.election_id = r.election_id
            WHERE r.receipt_hash = ?""", (_receipt_hash(code),)).fetchone()
    return dict(row) if row else None


def receipt_count(election_id):
    with get_connection() as connection:
        return connection.execute("SELECT COUNT(*) FROM receipts WHERE election_id = ?", (election_id,)).fetchone()[0]


# --------------------------------------------------------------------------
# Statistics for charts
# --------------------------------------------------------------------------

def turnout_timeline(election, buckets=24):
    """Cumulative votes over time for an election (times only, never choices)."""
    with get_connection() as connection:
        times = [row[0] for row in connection.execute(
            "SELECT created_at FROM ballots WHERE election_id = ? ORDER BY ballot_id", (election["election_id"],))]
    start = election["opened_at"] or (times[0] if times else None)
    if not start:
        return []
    from datetime import datetime as dt
    fmt = "%Y-%m-%d %H:%M:%S"
    t0 = dt.strptime(start[:19], fmt)
    t1 = dt.strptime((election["closed_at"] or now())[:19], fmt)
    span = max((t1 - t0).total_seconds(), 60)
    step = span / buckets
    points, count, i = [], 0, 0
    parsed = [dt.strptime(t[:19], fmt) for t in times]
    for b in range(buckets + 1):
        edge = t0.timestamp() + b * step
        while i < len(parsed) and parsed[i].timestamp() <= edge:
            count += 1
            i += 1
        label = "%H:%M:%S" if span < 1800 else "%H:%M" if span < 86400 else "%d %b %H:%M"
        points.append({"t": dt.fromtimestamp(edge).strftime(label), "votes": count})
    points[-1]["votes"] = len(parsed)
    return points


def registration_stats(days=14):
    from datetime import date, timedelta
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT substr(submitted_at, 1, 10) AS day, status, reviewed_by, dob, gender FROM registrations"
        ).fetchall()
        voters = connection.execute("SELECT COUNT(*) FROM voters").fetchone()[0]
        election = current_election(connection)
        voted = connection.execute("SELECT COUNT(*) FROM turnout WHERE election_id = ?",
                                   (election["election_id"],)).fetchone()[0]
    today = date.today()
    labels = [(today - timedelta(days=d)).isoformat() for d in range(days - 1, -1, -1)]
    series = {key: dict.fromkeys(labels, 0) for key in ("auto", "manual", "pending", "rejected")}
    ages = {"18-21": 0, "22-25": 0, "26-35": 0, "36-50": 0, "51+": 0}
    genders = {}
    for r in rows:
        key = ("auto" if r["reviewed_by"] == AUTO_APPROVER else "manual") if r["status"] in ("approved", "removed") \
            else r["status"]
        if r["day"] in series.get(key, {}):
            series[key][r["day"]] += 1
        if r["status"] == "approved":
            born = date.fromisoformat(r["dob"])
            age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
            bucket = "18-21" if age <= 21 else "22-25" if age <= 25 else "26-35" if age <= 35 else "36-50" if age <= 50 else "51+"
            ages[bucket] += 1
            genders[r["gender"]] = genders.get(r["gender"], 0) + 1
    submitted = len(rows)
    approved = sum(1 for r in rows if r["status"] in ("approved", "removed"))
    auto = sum(1 for r in rows if r["reviewed_by"] == AUTO_APPROVER)
    return {
        "days": [d[5:] for d in labels],
        "series": {k: list(v.values()) for k, v in series.items()},
        "ages": ages,
        "genders": genders,
        "funnel": {"applications": submitted, "approved": approved, "auto_approved": auto,
                   "voters": voters, "voted": voted},
    }


def log_booth_event(event, details=""):
    with transaction() as connection:
        log_event(connection, "booth", event, details)


def turnout_summary(election):
    with get_connection() as connection:
        voted = connection.execute(
            "SELECT COUNT(*) FROM turnout WHERE election_id = ?", (election["election_id"],)
        ).fetchone()[0]
        registered = election["eligible_count"]
        if registered is None:
            registered = connection.execute("SELECT COUNT(*) FROM voters").fetchone()[0]
    return {
        "registered": registered,
        "voted": voted,
        "remaining": max(registered - voted, 0),
        "participation": round(voted / registered * 100, 1) if registered else 0,
    }


def election_results(election):
    """Vote counts per candidate plus integrity checks. Call only for closed elections."""
    eid = election["election_id"]
    names = get_candidates(eid)
    with get_connection() as connection:
        counts = dict(connection.execute(
            "SELECT candidate, COUNT(*) FROM ballots WHERE election_id = ? GROUP BY candidate", (eid,)
        ).fetchall())
        turnout_count = connection.execute(
            "SELECT COUNT(*) FROM turnout WHERE election_id = ?", (eid,)
        ).fetchone()[0]

    values = [counts.get(name, 0) for name in names]
    total = sum(values)
    percentages = [round(v / total * 100, 1) if total else 0 for v in values]

    contenders = [(v, n) for n, v in zip(names, values) if n != NOTA]
    top = max((v for v, _ in contenders), default=0)
    leaders = [n for v, n in contenders if v == top] if top > 0 else []

    chain_ok, chain_message, _ = verify_ballot_chain()
    return {
        "labels": names,
        "values": values,
        "percentages": percentages,
        "total": total,
        "leaders": leaders,
        "tie": len(leaders) > 1,
        "nota_votes": counts.get(NOTA, 0),
        "checks": {
            "receipts": receipt_count(eid),
            "chain_ok": chain_ok,
            "chain_message": chain_message,
            "counts_match": total == turnout_count,
            "turnout_count": turnout_count,
        },
    }


def audit_entries(limit=300):
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM audit_log ORDER BY log_id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


if __name__ == "__main__":
    initialize_database()
    print("Database ready:", current_election())
