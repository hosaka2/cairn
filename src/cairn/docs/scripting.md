# スクリプトを書く

cairn が固定するのは **置き場所・export 名・最低限の形** だけです。取り込みと評価のロジックは自由。
書いたら `cairn check` で契約を検査できます（登録前に落ちるので、原因が分からず悩まずに済む）。

```bash
cairn init                        # datasets/ evals/ を作る
cairn new dataset sensor-scans    # schema.yaml + ingest.py
cairn new eval scan-anomaly       # table.yaml + run.py + v1.py
cairn check                       # 契約を検査（まずこれ）
```

## ディレクトリ規約

```
datasets/<name>/
  schema.yaml     列の定義（データセット名は <name> と一致させる）
  input.py        class Input(BaseModel) — フォームの項目。**読まれるだけで import されない**
  ingest.py       INGESTOR = <class>
evals/<name>/
  table.yaml      一覧に出す列の定義
  config.py       class Config(BaseModel) — 実行フォームの項目。同じく読まれるだけ
  run.py          RUNNER / PROCESS_FACTORY / CONFIG
  v1.py, v2.py …  EVALUATOR = <class>（ファイル名 = 評価方法の版）
```

重い依存は `ingest.py` / `run.py` に置きます。フォームを描く web はそれらを import せず、`input.py` / `config.py` を**読むだけ**です（規約ではなく、実行する経路が無い）。両者が同じ項目を名乗っているかは `cairn check` が見ます。

起動時にこの 2 ディレクトリを走査して自動登録します（`CAIRN_SCRIPTS`、既定はカレント）。
評価版として読むのは **`v` + 数字**（`v1.py` `v2.py`）のファイルだけです。

---

## schema.yaml（データセットの列）

```yaml
name: sensor-scans        # datasets/<name>/ と一致
kind: sensor              # 種別の表示ラベル（任意）
description: 説明
key: sample_id            # 行の identity。同じ key の再追記は上書き（後勝ち）
columns:
  - {name: sample_id,    type: str, required: true}
  - {name: site,         type: str}
  - {name: calibrated,   type: bool}
  - {name: waveform_url, type: str}     # 重いアセットは参照だけ持つ
  - {name: gt,           type: int, required: true}
nested:                               # 一覧に出ない子テーブル（時系列など）
  readings:
    schema:
      - {name: t,     type: float}
      - {name: value, type: float}
```

- `type`: `str` / `int` / `float` / `bool` / `datetime` / `s3path` / `list[float]` / `list[int]` /
  `list[str]` / `json`
- `required: true` の列が欠けた行は取り込みで却下されます。
- **重いもの（生波形・画像・マスク・点群）は台帳に入れない。** 列に URL/パスを持たせ、評価側で
  `ctx.read_bytes(url)` で読みます。列から**派生できる値は持たない**のが原則。
- `nested` は行の中にそのまま入り、評価時に `ctx.dataset.frames(sample_id)` で取れます。

## table.yaml（評価一覧の列）

```yaml
name: scan-anomaly
dataset: sensor-scans   # 既定の対象（実行フォームでプリセット・実行時に変更可）
columns:
  - {name: f1,        type: float, display: "F1", primary: true, direction: higher, scale: [0, 1]}
  - {name: miss_rate, type: float, display: "見逃し%", direction: lower}
  - {name: coverage,  type: float, display: "被覆%"}
default_sort: created_at desc
```

- `primary: true` の列が**主要指標**＝石の幅（1 列だけ。省略時は先頭列）。
- `direction`: `higher`（大きいほど良い）/ `lower`（誤差など小さいほど良い）。比較の色と石幅の向き。
- `scale: [min, max]`: 石幅の絶対レンジ。付けると僅差が誇張されません。
- **評価器が返す `row` はこの列と厳密一致**（余分な列はエラー）。`eval_id` / `snapshot_id` /
  `evaluator_version` などの共通メタは cairn が自動で付けます。

---

## Ingestor（取り込み）

```python
# datasets/sensor-scans/ingest.py
from pydantic import BaseModel
from cairn.interfaces.ingestor import IngestContext

class Ingest:
    class Input(BaseModel):        # ← ここから UI の入力欄が自動生成される
        source: str = ""

    execution = "inline"

    def ingest(self, inp: "Ingest.Input", ctx: IngestContext):
        raw = ctx.read_text(inp.source)          # s3:// gs:// file:// http(s):// を同じ API で
        for line in raw.splitlines():
            yield {"sample_id": ..., "site": ...}      # schema.yaml 準拠の dict を yield

INGESTOR = Ingest
```

**契約**: `INGESTOR` にクラスを割り当てる / `Input` は pydantic の `BaseModel` /
`ingest(self, inp, ctx)` が schema 準拠の dict を yield する。

`ctx` で使えるもの: `read_text` `read_bytes` `open`（backend 非依存。boto3 不要）、
`dataset` `tmpdir` `created_by`。Postgres 等の外部接続はスクリプトが自分の依存で行います。

## Runner / process_one（推論）

```python
# evals/scan-anomaly/run.py
import json
from pydantic import BaseModel
from cairn.core.records import RunSpec

class Config(BaseModel):           # ← 実行フォームの「設定」欄。実行時に変更できる
    threshold: float = 0.5
    model_config = {"extra": "ignore"}

def make_process_one(rows_by_id):
    def process_one(sample_id: str, run_config: dict) -> bytes | None:
        cfg = Config(**run_config)
        row = rows_by_id[sample_id]
        if 対象外(row):
            return None                       # ← 予測なし＝評価対象外（落とさない）
        return json.dumps({"pred": ...}).encode()
    return process_one

class Runner:
    class Config(BaseModel):
        chunk_size: int = 16
        model: dict = {}

    def plan(self, sample_ids: list[str], cfg: "Runner.Config") -> list[RunSpec]:
        return [RunSpec(sample_ids=sample_ids[i:i + cfg.chunk_size], job_name="infer",
                        run_config=cfg.model)
                for i in range(0, len(sample_ids), cfg.chunk_size)]

RUNNER = Runner
PROCESS_FACTORY = make_process_one
CONFIG = Config
```

**契約**: `RUNNER` は `Config`(pydantic) と `plan(sample_ids, cfg) -> list[RunSpec]` を持つ /
`PROCESS_FACTORY` は `process_one(sample_id, run_config) -> bytes | None` を返す関数 /
`CONFIG` は pydantic モデル。

- web から実行する場合、`RunSpec.job_name` は **`"infer"`** にしてください。
- 予測の中身は**不透明なバイト列**。形式は自由（JSON でも parquet でも）。読むのは評価器だけです。
- **`None` を返すと予測を書きません**。「その対象は成果物が無い」を落とさずに表現できます。

## Evaluator（評価）

```python
# evals/scan-anomaly/v1.py
from cairn.core.records import EvalResult, Metric

class Eval:
    def score(self, ctx) -> EvalResult:
        gt = {r["sample_id"]: r for r in ctx.dataset.rows()}
        total = 0.0
        for sample_id, data in ctx.predictions.iter():     # (id, bytes)
            ...
        return EvalResult(
            row={"f1": 0.882, "miss_rate": 6.3, "coverage": 98.3},      # table.yaml と一致
            metrics=[Metric(name="f1", value=0.882),
                     Metric(name="f1", value=0.0, sample_id="A_014")],
            report_md="## 結果\n\n…",              # 画面に出る詳細
            assets={"plot.svg": b"<svg .../>"},    # report.md から ![](assets/plot.svg) で参照
            metadata={"by_site": {"A": 0.88}},     # 列にしない自由 JSON
        )

EVALUATOR = Eval
```

**契約**: `EVALUATOR` にクラスを割り当てる / `score(self, ctx) -> EvalResult` /
`Config`(pydantic) は任意（あれば実行の `config` が入って渡る）。

`ctx` の中身:

| | 意味 |
|---|---|
| `ctx.dataset.rows()` | snapshot 時点の全行（GT） |
| `ctx.dataset.row(id)` / `targets()` | 1 行 / id 一覧 |
| `ctx.dataset.frames(id)` | `nested` の子テーブル（時系列など） |
| `ctx.predictions.iter()` | `(sample_id, bytes)` を順に |
| `ctx.config` | 実行時の設定（`Config` 型） |
| `ctx.expected_n` / `ctx.actual_n` | 対象数 / 予測が書かれた数（カバレッジ） |
| `ctx.read_bytes(url)` 等 | 参照アセット（生波形・画像など）の読取 |

守ること:

- **入力は ctx だけ**。隠れた可変状態やグローバル副作用を持たない（同じ入力なら同じ結果）。
  参照アセットを `ctx` 経由で読むのは正規の入力経路です。
- **分解できない指標は評価時に集計する**。mAP / AUC / 加重平均は全予測が揃って初めて決まるので、
  推論側で先に潰さず、ここで計算します。
- **マージした版ファイルは編集しない**。式を直すときは `cairn new eval-version <name> v2` で
  新しい版を切ります（過去の実行の意味を保つため。版が違う実行は並べて比較しません）。

---

## よくある失敗

| 症状 | 原因 | 直し方 |
|---|---|---|
| 実行フォームに「未登録」と出る | export 名の綴り違い、import エラー | `cairn check` で理由を見る |
| 入力欄が出ない | `Input` が pydantic でない | `class Input(BaseModel)` にする |
| 評価が `row が table.yaml に違反` で失敗 | 列の増減・型違い | `table.yaml` と `row` を一致させる |
| 石の幅が想定と逆 | `direction` 未設定（既定 higher） | 誤差系の列に `direction: lower` |
| 実験を比べられない | `snapshot_id` が違う | 台帳は凍結して実行だけ足す（値の変更・削除は snapshot を変える） |
| `v2.py` が読まれない | ファイル名が `v` + 数字でない | `v2.py` のように付ける |
