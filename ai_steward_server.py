"""
AI Steward — Connector Registry (Postgres Auto-Discover + S3 Prefix Scan)
========================================================================

This solution uses the MCP protocol to expose data discovery and scan tools, while delegating
higher-level tasks like documentation and remediation to an integrated AI Agent powered by an LLM.
"""
from mcp.server.fastmcp import FastMCP
from typing import Dict, Any, List
import pandas as pd
import os, io
from datetime import datetime
import re
import uuid
import boto3
import psycopg2
from psycopg2 import sql

mcp = FastMCP("MCP-AI-Steward-ConnectorsRegistry")

# ======================================
# CONNECTORS registry (POC — hardcoded)
# ======================================
CONNECTORS: Dict[str, Dict[str, Any]] = {
    "s3_data": {
        "type": "s3",
        "bucket": "aws-raw-layer-elite",
        "prefix": "customers/",
        "region": "us-west-2",
        # POC ONLY: hardcoded keys (replace before production!)
        "aws_access_key_id": "your-key",
        "aws_secret_access_key": "your-secret"
    },
    "pg_reporting": {
        "type": "postgres",
        # Example DSN: postgresql://USER:PASS@HOST:PORT/DB
        "url": os.getenv("PG_REPORTING_URL", "postgresql://postgres:test123@localhost:5433/reporting")
    }
}

# ==========================
# Internal stores and sample
# ==========================
DATASETS: Dict[str, Dict[str, Any]] = {}
CONNECTOR_RESULTS: Dict[str, Dict[str, Any]] = {}
AUDIT_LOG: List[Dict[str, Any]] = []

SAMPLE_CSV = 'sample_data.csv'
SAMPLE_DATA = '''
user_id,full_name,email,signup_date,country,age,phone
1,Alice Johnson,alice.j@example.com,2024-05-01,US,34,555-123-4567
2,Bob Kumar,bob.kumar@example.co.in,2024-06-12,IN,29,9876543210
3,Chen Wei,chen.wei@example.cn,2024-07-03,CN,41,
4,Diego Martínez,diego.m@example.mx,2024-04-21,MX,25,442-555-0199
5,Eva Müller,eva.mueller@example.de,2024-08-30,DE,38,491512345678
'''


def ensure_sample_csv():
    if not os.path.exists(SAMPLE_CSV):
        with open(SAMPLE_CSV, 'w', encoding='utf-8') as f:
            f.write(SAMPLE_DATA.strip() + '\n')
    DATASETS['sample'] = {
        'source': 'local_csv',
        'connector': 'local',
        'path': SAMPLE_CSV,
        'ingested_at': datetime.utcnow().isoformat()
    }

# ===============
# Small utilities
# ===============

def now_iso() -> str:
    return datetime.utcnow().isoformat()


def audit(event: str, details: Dict[str, Any]):
    entry = {'id': str(uuid.uuid4()), 'time': now_iso(), 'event': event, 'details': details}
    AUDIT_LOG.append(entry)
    return entry


# ===============================
# Discovery from Postgres & S3
# ===============================

def discover_postgres(connector_id: str) -> List[str]:
    cfg = CONNECTORS[connector_id]
    url = cfg['url']
    ds_ids: List[str] = []
    conn = psycopg2.connect(url)
    try:
        with conn.cursor() as cur:
            # exclude system schemas; list all base tables
            cur.execute(
                """
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                AND table_schema NOT IN ('pg_catalog','information_schema')
                ORDER BY table_schema, table_name;
                """
            )
            for schema, table in cur.fetchall():
                dataset_id = f"{connector_id}.{schema}.{table}"
                DATASETS[dataset_id] = {
                    'source': 'postgres',
                    'connector': connector_id,
                    'pg': {
                        'url': url,
                        'schema': schema,
                        'table': table
                    },
                    'ingested_at': now_iso()
                }
                ds_ids.append(dataset_id)
    finally:
        conn.close()
    audit('discover_postgres', {'connector': connector_id, 'datasets': ds_ids})
    return ds_ids


def discover_s3(connector_id: str) -> List[str]:
    cfg = CONNECTORS[connector_id]
    session_kwargs = {}
    if 'region' in cfg: session_kwargs['region_name'] = cfg['region']
    if 'aws_access_key_id' in cfg and 'aws_secret_access_key' in cfg:
        session_kwargs['aws_access_key_id'] = cfg['aws_access_key_id']
        session_kwargs['aws_secret_access_key'] = cfg['aws_secret_access_key']
    s3 = boto3.client('s3', **session_kwargs)

    bucket = cfg['bucket']
    prefix = cfg.get('prefix', '')

    ds_ids: List[str] = []
    continuation = None
    while True:
        kwargs = {'Bucket': bucket, 'Prefix': prefix}
        if continuation:
            kwargs['ContinuationToken'] = continuation
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get('Contents', []):
            key = obj['Key']
            if key.endswith('/'):
                continue
            # treat each object as a dataset, only CSV for now
            if not key.lower().endswith('.csv'):
                continue
            dataset_id = f"{connector_id}:{key}"
            DATASETS[dataset_id] = {
                'source': 's3',
                'connector': connector_id,
                's3': {
                    'bucket': bucket,
                    'key': key
                },
                'ingested_at': now_iso()
            }
            ds_ids.append(dataset_id)
        if resp.get('IsTruncated'):
            continuation = resp.get('NextContinuationToken')
        else:
            break
    audit('discover_s3', {'connector': connector_id, 'datasets': ds_ids})
    return ds_ids


def _discover_all_internal() -> List[str]:
    discovered: List[str] = []
    for cid, spec in CONNECTORS.items():
        if spec['type'] == 'postgres':
            discovered += discover_postgres(cid)
        elif spec['type'] == 's3':
            discovered += discover_s3(cid)
    return discovered

# =====================
# Data access (to DF)
# =====================

def read_df(dataset_id: str, sample_limit: int = 1000) -> pd.DataFrame:
    ds = DATASETS.get(dataset_id)
    if not ds:
        raise ValueError("dataset not found")
    src = ds['source']
    if src == 'local_csv':
        return pd.read_csv(ds['path'])
    if src == 'postgres':
        pg = ds['pg']
        conn = psycopg2.connect(pg['url'])
        try:
            with conn.cursor() as cur:
                q = sql.SQL("SELECT * FROM {}.{} LIMIT {};").format(
                    sql.Identifier(pg['schema']), sql.Identifier(pg['table']), sql.Literal(sample_limit)
                )
                cur.execute(q)
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
                return pd.DataFrame(rows, columns=cols)
        finally:
            conn.close()
    if src == 's3':
        s3meta = ds['s3']
        cfg = CONNECTORS[ds['connector']]
        session_kwargs = {}
        if 'region' in cfg: session_kwargs['region_name'] = cfg['region']
        if 'aws_access_key_id' in cfg and 'aws_secret_access_key' in cfg:
            session_kwargs['aws_access_key_id'] = cfg['aws_access_key_id']
            session_kwargs['aws_secret_access_key'] = cfg['aws_secret_access_key']
        s3 = boto3.client('s3', **session_kwargs)
        obj = s3.get_object(Bucket=s3meta['bucket'], Key=s3meta['key'])
        body = obj['Body'].read()
        return pd.read_csv(io.BytesIO(body))
    raise ValueError(f"unknown source: {src}")

# ==========================
# Connectors (DF processors)
# ==========================
class MetadataExtractor:
    name = 'metadata_extractor'
    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        schema = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            samples = df[col].dropna().astype(str).head(5).tolist()
            schema.append({'column': str(col), 'dtype': dtype, 'sample_values': samples, 'nullable': bool(df[col].isna().any())})
        return {'rows': int(len(df)), 'columns': int(len(df.columns)), 'schema': schema}

class DataClassifier:
    name = 'data_classifier'
    KEYWORDS = {
        'personal': ['name', 'full_name', 'first_name', 'last_name', 'username'],
        'contact': ['email', 'phone', 'mobile', 'contact'],
        'id': ['id', 'user_id', 'uuid'],
        'geo': ['country', 'city', 'state', 'zip']
    }
    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        import re as _re
        tags = {}
        suggestions = {}
        for col in df.columns:
            col_lower = str(col).lower()
            col_tags = set()
            for tag, kwlist in self.KEYWORDS.items():
                for kw in kwlist:
                    if kw in col_lower:
                        col_tags.add(tag)
            if _re.search(r'email', col_lower):
                col_tags.add('email')
            if _re.search(r'phone|mobile', col_lower):
                col_tags.add('phone')
            if _re.search(r'(^|_)id($|_)', col_lower):
                col_tags.add('identifier')
            tags[str(col)] = list(col_tags) if col_tags else ['unknown']
            suggestions[str(col)] = f"Consider tagging as: {', '.join(tags[str(col)])}"
        return {'tags': tags, 'suggestions': suggestions}

class QualityMonitor:
    name = 'quality_monitor'
    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        checks = {}
        total = len(df)
        for col in df.columns:
            nulls = int(df[col].isna().sum())
            null_pct = round(nulls / total * 100, 2) if total else None
            unique = int(df[col].nunique(dropna=True))
            checks[str(col)] = {'null_count': nulls, 'null_pct': null_pct, 'unique_values': unique}
        duplicate_rows = int(df.duplicated().sum())
        health_score = 100 - sum(v['null_pct'] for v in checks.values())/len(checks) if checks else 100
        return {'column_checks': checks, 'duplicate_rows': duplicate_rows, 'health_score': round(health_score,2)}

class ComplianceEnforcer:
    name = 'compliance_enforcer'
    EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\d{10}|\d{3}[-.\s]\d{3}[-.\s]\d{4})\b")
    AADHAAR_RE = re.compile(r"\b\d{12}\b")
    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        flags = {}
        for col in df.columns:
            col_vals = df[col].dropna().astype(str)
            sample_text = ' '.join(col_vals.head(50).tolist())
            has_email = bool(self.EMAIL_RE.search(sample_text))
            has_phone = bool(self.PHONE_RE.search(sample_text))
            has_aadhaar = bool(self.AADHAAR_RE.search(sample_text))
            flags[str(col)] = {'has_email': has_email, 'has_phone': has_phone, 'has_aadhaar_like': has_aadhaar}
        policy_actions = {c: (['mask', 'restricted_access'] if any(flags[c].values()) else []) for c in flags}
        return {'pii_flags': flags, 'policy_suggestions': policy_actions}

CONNECTOR_IMPLS = [MetadataExtractor(), DataClassifier(), QualityMonitor(), ComplianceEnforcer()]

# ==========================
# Connector execution
# ==========================

def run_connectors_impl(dataset_id: str) -> Dict[str, Any]:
    df = read_df(dataset_id)
    results = {}
    for c in CONNECTOR_IMPLS:
        results[c.name] = c.run(df)
    CONNECTOR_RESULTS[dataset_id] = results
    audit('connectors_run', {'dataset_id': dataset_id, 'connectors': list(results.keys())})
    return {'status': 'done', 'dataset_id': dataset_id, 'connectors': list(results.keys())}

# ==========================
# MCP Resources
# ==========================
@mcp.resource("connectors://list")
def connectors_list() -> Dict[str, Any]:
    # redact secrets in output
    def scrub(v: Dict[str, Any]) -> Dict[str, Any]:
        redacted = dict(v)
        for k in list(redacted.keys()):
            if 'secret' in k or 'access_key' in k:
                redacted[k] = '***'
        return redacted
    return {k: scrub(v) for k, v in CONNECTORS.items()}


@mcp.resource("datasets://list")
def datasets_list() -> Dict[str, Any]:
    return DATASETS


@mcp.resource("datasets://{dataset_id}/schema")
def dataset_schema(dataset_id: str) -> Dict[str, Any]:
    cr = CONNECTOR_RESULTS.get(dataset_id)
    if cr and 'metadata_extractor' in cr:
        return cr['metadata_extractor']
    df = read_df(dataset_id)
    return MetadataExtractor().run(df)


@mcp.resource("datasets://{dataset_id}/sample_rows")
def sample_rows(dataset_id: str) -> List[Dict[str, Any]]:
    df = read_df(dataset_id)
    return df.head(5).to_dict(orient='records')


@mcp.resource("datasets://{dataset_id}/classification")
def classification(dataset_id: str) -> Dict[str, Any]:
    cr = CONNECTOR_RESULTS.get(dataset_id)
    if cr and 'data_classifier' in cr:
        return cr['data_classifier']
    df = read_df(dataset_id)
    return DataClassifier().run(df)


@mcp.resource("datasets://{dataset_id}/quality")
def quality(dataset_id: str) -> Dict[str, Any]:
    cr = CONNECTOR_RESULTS.get(dataset_id)
    if cr and 'quality_monitor' in cr:
        return cr['quality_monitor']
    df = read_df(dataset_id)
    return QualityMonitor().run(df)


@mcp.resource("datasets://{dataset_id}/compliance")
def compliance(dataset_id: str) -> Dict[str, Any]:
    cr = CONNECTOR_RESULTS.get(dataset_id)
    if cr and 'compliance_enforcer' in cr:
        return cr['compliance_enforcer']
    df = read_df(dataset_id)
    return ComplianceEnforcer().run(df)


@mcp.resource("datasets://{dataset_id}/connector_results")
def connector_results(dataset_id: str) -> Dict[str, Any]:
    return CONNECTOR_RESULTS.get(dataset_id, {})


@mcp.resource("audit://log")
def audit_log() -> List[Dict[str, Any]]:
    return AUDIT_LOG

# ==========================
# Steward helpers & aggregate resources
# ==========================

# Compute a compact quality summary for a dataset (runs QualityMonitor if needed)
def _quality_summary(dataset_id: str) -> Dict[str, Any]:
    cr = CONNECTOR_RESULTS.get(dataset_id)
    if not cr or 'quality_monitor' not in cr:
        run_connectors_impl(dataset_id)
        cr = CONNECTOR_RESULTS.get(dataset_id, {})
    qm = cr.get('quality_monitor', {})
    health = qm.get('health_score')
    null_hotspots = {c: v for c, v in (qm.get('column_checks') or {}).items() if v.get('null_pct', 0) >= 20}
    return {
        'dataset_id': dataset_id,
        'health_score': health,
        'duplicate_rows': qm.get('duplicate_rows'),
        'null_hotspots': null_hotspots
    }

@mcp.resource("datasets://{dataset_id}/health")
def dataset_health(dataset_id: str) -> Dict[str, Any]:
    return _quality_summary(dataset_id)

@mcp.resource("steward://quality_ranking")
def quality_ranking() -> List[Dict[str, Any]]:
    # rank datasets from worst to best (lowest health first)
    summaries = []
    for dsid in DATASETS.keys():
        try:
            summaries.append(_quality_summary(dsid))
        except Exception:
            continue
    summaries = [s for s in summaries if s.get('health_score') is not None]
    return sorted(summaries, key=lambda x: x['health_score'])

# ==========================
# MCP Tools
# ==========================
@mcp.tool()
def discover_all() -> Dict[str, Any]:
    found = _discover_all_internal()
    return {"discovered": found}


@mcp.tool()
def scan_connector(connector_id: str) -> Dict[str, Any]:
    """Scan a connector by id (pg_reporting/s3_data) OR a single dataset id as a convenience.
    Returns scanned datasets plus a quality_ranking snippet.
    """
    ds: List[str] = []
    if connector_id in CONNECTORS:
        spec = CONNECTORS[connector_id]
        if spec['type'] == 'postgres':
            ds = discover_postgres(connector_id)
        elif spec['type'] == 's3':
            ds = discover_s3(connector_id)
    elif connector_id in DATASETS:
        ds = [connector_id]
    else:
        return {"status": "error", "error": "connector_or_dataset_not_found", "id": connector_id}

    summaries = []
    for d in ds:
        run_connectors_impl(d)
        try:
            summaries.append(_quality_summary(d))
        except Exception as e:
            summaries.append({"dataset_id": d, "error": str(e)})
    ranked = [s for s in summaries if s.get('health_score') is not None]
    ranked = sorted(ranked, key=lambda x: x['health_score'])
    return {"status": "done", "scanned": ds, "quality_ranking": ranked[:10]}


@mcp.tool()
def get_sample_rows(dataset_id: str, limit: int = 5) -> List[Dict[str, Any]]:
    df = read_df(dataset_id)
    return df.head(limit).to_dict(orient='records')


@mcp.tool()
def find_rows(dataset_id: str, column: str, pattern: str, limit: int = 5) -> List[Dict[str, Any]]:
    df = read_df(dataset_id)
    if column not in df.columns:
        return []
    import re as _re
    regex = _re.compile(pattern)
    matched = df[df[column].astype(str).str.match(regex, na=False)]
    return matched.head(limit).to_dict(orient='records')


# =====================
# Bootstrap
# =====================
ensure_sample_csv()
# initial discover so user can start right away
try:
    _ = _discover_all_internal()
except Exception as e:
    audit('discover_error', {'error': str(e)})

if __name__ == '__main__':
    print('Run with: uv run mcp dev ai_steward_server_connectors_registry.py')
