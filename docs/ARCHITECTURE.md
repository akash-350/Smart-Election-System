# Smart Election System: Architecture

A classroom demonstration of face-verified electronic voting, built with Flask, OpenCV and SQLite.

## 1. System overview

```mermaid
flowchart LR
    subgraph Browser
        A[Admin pages<br/>dashboard, election setup,<br/>voters, audit, accuracy]
        B[Voting booth<br/>camera + ballot]
        R[Results declaration]
    end

    subgraph Flask app
        AUTH[auth.py<br/>admin login, lockout,<br/>JSON-only actions]
        APP[app.py<br/>admin routes + APIs]
        VOTE[voting_routes.py<br/>identify, confirm, cast]
    end

    subgraph Core
        FACE[face_model.py<br/>Haar face + eye detection,<br/>KNN matching, liveness,<br/>accuracy test]
        DB[database.py<br/>elections, ballots, audit,<br/>hash chains]
    end

    subgraph Storage[data/]
        PKL[(faces_data.pkl<br/>names.pkl)]
        PH[(photos/*.jpg)]
        SQL[(voting_demo.db<br/>SQLite + triggers)]
    end

    A --> AUTH --> APP
    B --> AUTH --> VOTE
    R --> APP
    APP --> FACE
    APP --> DB
    VOTE --> FACE
    VOTE --> DB
    FACE --> PKL
    FACE --> PH
    DB --> SQL
```

The desktop scripts `add_faces.py` (registration) and `give_vote.py` (OpenCV voting window) use the same `face_model.py` and `database.py`.

## 2. Voter self-registration

```mermaid
flowchart TD
    A[Voter opens /register] --> B[Details: name, DOB, gender,<br/>Voter ID or Aadhaar, mobile,<br/>ID card photo, consent]
    B --> C[Face capture + close/open eyes]
    C --> D{Automatic checks}
    D -->|ID format / Verhoeff fails,<br/>ID already used, under 18,<br/>face unclear, liveness fails| X[Refused immediately<br/>nothing stored]
    D -->|passes, maybe with warnings:<br/>possible duplicate face,<br/>low quality, no photo on card| P[Stored as Pending<br/>reference SE-0001]
    P --> R{Officer reviews:<br/>live photo vs ID card,<br/>check results}
    R -->|Approve| V[Added to voter list,<br/>face added to recogniser]
    R -->|Reject + reason| J[Rejected; face and ID card<br/>images deleted]
    V --> S[Voter checks status with<br/>reference + date of birth]
    J --> S
```

| Check | Type | Catches |
|---|---|---|
| EPIC format `AAA0000000` / Aadhaar 12 digits, not starting 0-1 | refuse | made-up or mistyped IDs |
| Aadhaar Verhoeff checksum | refuse | typos, random numbers |
| ID hash already pending/approved | refuse | the same ID used twice |
| Age 18+ | refuse | underage |
| ≥30 of 51 frames with exactly one clear face | refuse | bad capture |
| Liveness (eyes closed vs open) | refuse | printed photo |
| Face matches an existing voter or applicant | warn | one person, many IDs |
| Other faces in frame, low brightness or blur | warn | helpers in frame, poor photos |
| No face found on the ID card image | warn | wrong or unreadable document |

**ID privacy.** Full numbers are never stored: `id_masked` keeps the last 4 digits for display and
`id_hash = HMAC-SHA256(secret key, type:number)` detects duplicates. Without the key in
`data/.id_pepper`, the 12-digit Aadhaar space can't be brute-forced from the hash.

**Not possible in this demo:** confirming with UIDAI / Election Commission that an ID really exists
(needs official API access), and SMS OTP for the mobile number (needs a paid SMS gateway).

## 3. Election lifecycle

An election only moves forward. A closed election can never be reopened; a new election is created instead, and old ones stay in history.

```mermaid
stateDiagram-v2
    [*] --> Setup: election created
    Setup --> Setup: rename / add, rename, remove candidates
    Setup --> Open: officer opens voting\n(ballot locked)
    Open --> Open: voters cast ballots\n(results sealed)
    Open --> Closed: officer types election name to confirm
    Closed --> [*]: results declared,\nnew election may be started
```

## 4. Voting flow (one voter)

```mermaid
sequenceDiagram
    actor V as Voter
    participant B as Booth page
    participant S as Flask server
    participant F as face_model
    participant D as SQLite

    V->>B: Start face check
    B->>B: capture 5 frames (eyes open)
    B->>V: "Close your eyes" (text + voice)
    B->>B: capture 6 frames
    B->>V: "Open your eyes"
    B->>B: capture 4 frames
    B->>S: POST /api/identify-demo-voter (all frames)
    S->>F: identify_frames(open frames)
    F-->>S: voter ID (or rejected: unknown / uncertain)
    S->>F: check_liveness(closed vs open frames)
    F-->>S: alive? (a printed photo can't close its eyes)
    S->>D: registered? already voted?
    S-->>B: voter ID
    V->>B: "Yes, that's me"
    B->>S: POST /api/confirm-demo-voter
    V->>B: choose candidate, Cast my vote
    B->>S: POST /api/cast-demo-ballot
    S->>D: one transaction: add turnout row + append hash-chained ballot + audit "Ballot cast"
    S-->>B: recorded
    B->>V: Thank you (resets for next voter)
```

## 5. Data model

```mermaid
erDiagram
    ELECTIONS ||--o{ CANDIDATES : has
    ELECTIONS ||--o{ BALLOTS : receives
    ELECTIONS ||--o{ TURNOUT : records
    VOTERS ||--o{ TURNOUT : "voted in"
    REGISTRATIONS |o--o| VOTERS : "approved as"

    ELECTIONS {
        int election_id PK
        text name
        text state "setup | open | closed"
        text opened_at
        text closed_at
        int eligible_count
    }
    CANDIDATES {
        int candidate_id PK
        int election_id FK
        text name
    }
    VOTERS {
        text voter_id PK
        text registered_at
        text full_name
        text source "officer | self"
    }
    REGISTRATIONS {
        text ref PK "SE-0001"
        text full_name
        text dob
        text id_type "epic | aadhaar"
        text id_masked
        text id_hash "HMAC, never the number"
        text status "pending | approved | rejected | removed"
        text checks "JSON results"
        text duplicate_of
    }
    TURNOUT {
        int election_id PK
        text voter_id PK "WITHOUT ROWID: no insertion order"
    }
    BALLOTS {
        int ballot_id PK
        int election_id FK
        text candidate "no voter column"
        text created_at
        text prev_hash
        text hash
    }
    AUDIT_LOG {
        int log_id PK
        text created_at
        text actor
        text event
        text details
        text prev_hash
        text hash
    }
```

**Secret ballot.** `ballots` has no voter column. `turnout` records *who* voted but not *what*, and it is a `WITHOUT ROWID` table, so its storage order can't be matched against ballot order. The audit log records "Ballot cast" without a voter ID.

## 6. Integrity rules (enforced by SQLite triggers)

| Rule | How |
|---|---|
| Ballots can't be edited or deleted | `BEFORE UPDATE/DELETE` triggers raise an error |
| Ballots only while voting is open | `BEFORE INSERT` checks the election state |
| Only real candidates | `BEFORE INSERT` checks the candidates table |
| One vote per voter per election | `turnout` primary key + check in the same transaction |
| Turnout and audit log are permanent | update/delete triggers |
| Election only moves setup → open → closed | `BEFORE UPDATE` trigger on state |
| Ballot locked after opening | candidate triggers check `state = 'setup'` |
| Voters who voted can't be removed | `BEFORE DELETE` trigger + foreign key |
| Tampering is detectable | each ballot and audit entry stores `hash = SHA-256(previous hash, contents)`; the results and audit pages re-verify the chain |
| Registration decisions are final | trigger: pending → approved/rejected only; records never deleted |
| Results sealed until close | the results API and CSV export refuse while open |

## 7. Face recognition

1. **Detect.** OpenCV Haar cascade (`haarcascade_frontalface_default.xml`) finds exactly one face.
2. **Normalise.** Crop to 50×50, convert to grayscale, equalise the histogram, subtract the mean, scale to unit length (reduces lighting effects).
3. **Match.** k-nearest neighbours (k = 5). Accept only if at least 4 of 5 neighbours are the same voter **and** their mean distance is below `FACE_MATCH_THRESHOLD` (default 0.85). Otherwise the face is rejected as unknown.
4. **Agree.** The booth requires most frames to give the same voter, and no frame to give a different one.
5. **Liveness.** Eyes are detected with `haarcascade_eye.xml`. Eyes must be seen clearly more often in the open frames than in the "close your eyes" frames.

The **Accuracy test** page measures genuine accept rate, false reject rate, misidentification and false accept rate, using leave-out evaluation on the stored samples.

## 8. Known limitations (for the report)

- Pixel-based KNN is far weaker than modern face embeddings; accuracy drops with different lighting, angles or days.
- The liveness check stops printed photos but not a video replayed on a phone.
- Haar eye detection can struggle with glasses; set `LIVENESS_CHECK=off` if needed.
- Hash chains detect tampering but can't prevent someone with full file access from rebuilding the whole chain; a real system would publish the latest hash or sign it.
- Single admin account; no role separation between the election officer and the auditor.

## Screenshots

| | |
|---|---|
| ![Dashboard](screenshots/dashboard.png) | ![Election setup](screenshots/election.png) |
| ![Voting booth](screenshots/booth.png) | ![Ballot in Hindi](screenshots/ballot_hindi.png) |
| ![Results](screenshots/results.png) | ![Audit log](screenshots/audit.png) |
| ![Accuracy](screenshots/accuracy.png) | ![Self-registration](screenshots/register.png) |
