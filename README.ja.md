# cairn

[English](README.md) | [日本語](README.ja.md)

> **cairn**（ケルン）— 石を積み上げた道標。追記で積み上がり、不変の記録として残り、後から来た人の道標になる。

**DB 不要・モダリティ非依存・オーケストレータ非依存**の評価/データセット・レジストリ。
LLM / 物体検出 / セグメンテーション / SfM / テーブルなど、異種の推論パイプラインの
**データセットと評価結果を、オブジェクトストレージだけで**追記・記録・比較する。

- 単一の真実は**オブジェクトストレージ**（S3 / GCS / ローカル）。DB サーバを立てない。
- 箱（プラットフォーム）が固定するのはスキーマ・命名・不変性だけ。**取り込みと評価のロジックはスクリプトの自由**。
- **ロックレス**: 書き込みは常に新キー・評価は snapshot でキー集合を固定。CAS もリースも不要（下記）。
- `file://` で **MinIO もオーケストレータも無し**にローカルで動く。

> 学習・モデルレジストリ（→ MLflow/W&B）、LLM トレース（→ Langfuse）、推論の実行（→ Dagster 等）は
> スコープ外。cairn は**評価とデータセットの台帳**に徹する。

## インストール

PyPI 上の `cairn` は無関係な別プロジェクトのため、当面はリポジトリから:

```bash
pip install git+https://github.com/hosaka2/cairn      # GCS 以外は全部入り
```

Python 3.10 以上。GCS を使う場合:
`pip install "cairn[gcs] @ git+https://github.com/hosaka2/cairn"`。

cairn 自体を触るとき:

```bash
uv sync --extra dev     # テスト・lint・型チェックの道具が入る
uv run pytest           # カバレッジも一緒に測り、100% を下回ると落ちる
uv run ruff check src tests
uv run pyright          # src と tests
```

カバレッジは分岐込みで測り、100% 未満は失敗にしている（到達しづらい行は、簡単にすべき行だという判断）。
`pyright` は standard モードで `src` と `tests` の両方を見る。

## 設定（`CAIRN_ROOT`）

解決順は **`--root` > 環境変数 > カレントの `.env`**。使いやすい方でよい:

```bash
cairn --root file:///tmp/cairn-demo dataset ls   # フラグ
export CAIRN_ROOT=s3://bucket/cairn               # 環境変数
```
```env
# .env（カレントに置くと自動読込）
CAIRN_ROOT=s3://bucket/cairn
AWS_PROFILE=myprofile        # or AWS_ACCESS_KEY_ID / SECRET、AWS_ENDPOINT_URL(MinIO)
CAIRN_LANG=ja                # マニュアルとフォーム説明を日本語に（未設定なら英語）
```

> S3/GCS の認証は fsspec（s3fs/gcsfs＝boto3 のクレデンシャルチェーン）を通す:
> 環境変数・`~/.aws`・IAM ロールを自動で拾う。DuckDB 読みも同じチェーン（credential_chain）。

## 1 分で試す（file:// ・外部サービス不要）

```bash
export CAIRN_ROOT=file:///tmp/cairn-demo    # または .env に書く / --root で渡す
cairn demo-seed            # 合成データで全機能を実際に動かす（下記）
cairn web                  # http://127.0.0.1:8000 を開く
```

**評価の実行が石のように積み上がり**（幅＝主要指標）、データセットは**気軽な追記**として充実していく。

## 画面

![評価一覧](assets/evals-list.png)

評価一覧。種類ごとに `table.yaml` の列で表示し、**実行が右に石として積み上がる**
（石の幅＝主要指標、色＝最新の実行／同じ評価方法／前の評価方法（並べて比較できない））。
実行数の下の `v2: 3` は、最新の実行を採点した方法とその件数。

> **主要指標（石の幅）は `table.yaml` で明示する**: 列に `primary: true`（1 つだけ／省略時は先頭列）。
> `direction: higher|lower` で良い向き（誤差 px は lower）、`scale: [min, max]` で絶対レンジ
> （付けると僅差が表示中の min/max で誇張されない）。

![実行とレポート](assets/eval-runs.png)

1 つの実行。上に行、続いてメモ、メタデータ（config ＋ どの評価スクリプトで採点したか）、そして
**評価スクリプトが出力したレポート**。レポートは Markdown なので表・画像・SVG はスクリプト次第。
`Open ↗` で全幅表示。

![実行の比較](assets/eval-compare.png)

`Compare with another run` で、今見ている実行の下に他の実行が並ぶ。差分は
**「この実行がその行に対してどうだったか」**で、列の `direction` に従って着色する。
`evaluator_version` が違う実行はそもそも候補に出さない。**データ（snapshot）が違う**実行は候補に出るが、
行に `other data` と印が付く — snapshot をまたいだ差分は同じ問いに答えていないため。

![データセット](assets/dataset-rows.png)

データセット。左が追記履歴（石の幅＝その追記で増えた行数）、右が行で、`schema.yaml` に従って表示する
（`bool` はタグ、など）。

UI から一周できる: **データセット作成**（schema.yaml）→ **データ追加**（Ingestor フォーム／JSONL・プレビュー→確定）
→ **評価テーブル作成**（table.yaml）→ **実行を発行**→（推論が予測を書く）→ **Evaluate** → 結果が石に積まれる。

![評価テーブルの作成](assets/create-eval.png)

YAML の入力フォームには、その YAML の説明と、今ある定義から生成した例を横に出す。

![データの追加](assets/add-data.png)

データ追加の入口は 2 つ: **ingest スクリプトの `Input` から生成したフォーム**と、JSONL の貼り付け。
どちらも 入力 → プレビュー → 確定 で、確定するまで何も書き込まない。

![評価の実行](assets/run-eval.png)

実行の発行では、データセットと評価バージョンを選び、`run.py` の `CONFIG` をこの実行用に編集する。
config は実行と一緒に保存され、モデルの同一性——重みの版とそのパラメータ——はここに載る。
**Run** は実行を作ってデータを凍らせるだけで、ここでは何も推論しない。

![発行された実行と、ジョブが必要とする eval_id](assets/eval-running.png)

発行された実行は **Running** に並び、ジョブが必要とする `eval_id` が見える。
推論する側は `cairn eval targets` で対象を取り、`cairn eval put-prediction` で書き戻す。
その版の評価器が登録されていれば **Evaluate** が出る。進捗はオーケストレータ側の話で、
台帳は「発行済みで結果がまだ無い」ことだけを持つ。データは発行時点で凍っているので、
その間にデータセットへ追記しても測定対象は動かない。

![ヘルプ](assets/help.png)

同梱マニュアルは `/help` で読める。横には、契約を満たさないスクリプトがあればそれが出る（`cairn check` と同じ内容）。

> **cairn は推論をしない。** 実行を発行し、返ってきたものを保持し、採点する。demo スクリプトをこのプロセスで
> 回すのは独立したコマンド（`cairn eval run`）で、小さいデータセットでスクリプトを試すときの道でもある。
> オーケストレータへ直接 submit する道（`OrchestratorAdapter`）は未配線。

### `demo-seed` が実際に通すもの（faked なし）

同梱の [`cairn/demo/`](src/cairn/demo/) は合成センサー異常検知パイプラインで、**4 つのインターフェースを本当に実行**する:

- **Ingestor**（`ingest.py`）: seed で決定的な合成データを生成 → 追記。加えて **upsert 補正・tombstone 削除**も実演。
- **Runner**（`model.py`）: サンプルをチャンクに束ねて `RunSpec` を計画。
- **OrchestratorAdapter**（InlineAdapter）: `process_one` をその場で実行し**予測を書く**。
- **Evaluator**（`evaluate.py`）: 予測 + GT を集計して **F1/適合率/再現率を評価時に計算**（＝サンプル分解不可の指標を正しく扱う）＋混同行列レポート + SVG。
- **evaluator_version v1/v2**: 同じ予測でも計算が違う（v1 は F1≒正解率の粗い版、v2 は正式 F1）→ **評価方法を跨ぐ比較が無意味**なことを実演。

メトリクスは config（しきい値）で実際に変わり、UI の石の幅・delta に反映される。

## CLI で回す（追記 → 実行を発行 → 評価）

```bash
export CAIRN_ROOT=file:///tmp/cairn-demo

# データセット: schema を作って JSONL を追記（版を意識せず・追記のみ）
cat > /tmp/s.yaml <<'YAML'
name: demo
key: id
columns:
  - {name: id, type: str, required: true}
  - {name: label, type: int, required: true}
YAML
cairn dataset create --schema /tmp/s.yaml
printf '{"id":"a","label":1}\n{"id":"b","label":0}\n' > /tmp/r.jsonl
cairn dataset ingest demo --jsonl /tmp/r.jsonl
cairn dataset show demo

# 評価: 実行を発行（snapshot 固定）→ 予測を書いて評価（予測と評価は分離）
cairn eval create-table --table table.yaml
cairn eval create-run <table> --dataset demo --evaluator-version v1 \
    --title "初回" --config '{"threshold": 0.5}'
cairn eval score <table> <eval_id> --evaluator my.module:MyEvaluator

# 推論を外（オーケストレータ）で回す場合: 実行の発行と予測の受け取りだけ cairn を通る
cairn eval targets <table> <eval_id>                       # サンプル id を 1 行 1 件で
cairn eval put-prediction <table> <eval_id> --jsonl preds.jsonl

cairn eval withdraw <table> <eval_id> --reason "重みが違った"      # 一覧から取り下げ（記録は残る）
cairn eval ls
cairn vacuum                      # 古い checkpoint を回収（安全）
```

> 規約スクリプト一式で試すなら `cairn demo-init`（デモを `datasets/` `evals/` に配線）→ `cairn demo-seed`。

Python から一気通貫で回す例は [`tests/test_smoke.py`](tests/test_smoke.py) を参照。

## マニュアル

同梱しているので、インストール先でも読めます:

```bash
cairn docs manual      # 使い方（操作・CLI・比較できる条件）
cairn docs scripting   # スクリプトを書く（契約とリファレンス）
cairn check            # 書いたスクリプトが契約どおりか検査
```

web からは右上の**ヘルプ**（`/help`）。実体は [`src/cairn/docs/`](src/cairn/docs/)。

## スクリプトの規約（迷わない・コマンドで生成）

「箱」が固定するのは**置き場所と export 名**。中身は自由。cairn は起動時に下記を走査して自動登録する
（`CAIRN_SCRIPTS`、既定=カレント）:

```
datasets/<name>/
  schema.yaml            列定義（一覧に出る列の形）
  ingest.py              INGESTOR = <Ingestor>        取り込み。Input から UI フォーム自動生成
evals/<name>/
  table.yaml             評価一覧の列定義
  run.py                 RUNNER / PROCESS_FACTORY / CONFIG   推論単位（本番と同じ process_one）
  v1.py, v2.py, …        EVALUATOR = <Evaluator>      評価器。マージ後は編集禁止（版を刻む）
```

生成コマンド（規約どおりに雛形が出る）:

```bash
cairn init                       # datasets/ evals/ を作る
cairn new dataset my-images --kind detection
cairn new eval my-detect         # v1.py まで生成
cairn new eval-version my-detect v2   # 評価方法の新版（旧版は触らない）
cairn demo-init                  # デモを規約ファイルとして配線（一緒に動くお手本）
```

雛形を書いた後、登録するとストレージ上にデータセット/テーブルができます:

```bash
cairn dataset create --schema datasets/my-images/schema.yaml
cairn eval create-table --table evals/my-detect/table.yaml
cairn check                      # スクリプトが契約どおりか検査
```

## 構造

```
src/cairn/
  core/        箱が固定する部分（storage/schema/dataset/evals/records/config）
  interfaces/  Ingestor / Runner / Evaluator（固定するのはこの3つ）
  adapters/    OrchestratorAdapter（local はその場で実行）
  cli/         typer（CLI と Web は同じ core を通る）
  web/         read-only + 書き込みフロー（FastAPI + Jinja2・light）
  registry.py  規約ディレクトリの discover（datasets/ evals/ を走査して登録）
  scaffold.py  cairn init / new … のテンプレート
  demo/        全インターフェースを実行する合成デモ（demo-init の配線先）
```

ストレージ配置は「データセット = 気軽な追記」「**評価 = 実行を積む石**」。

## コンセプト

- **2 軸で結果を刻む**: `(snapshot_id, evaluator_version)`。`snapshot_id` が一致する run どうしだけが
  比較可能（データ内容の同一性が自動で分かる）。`evaluator_version` を必ず刻むことで「先月 0.81 /
  今月 0.79」が中身の変化か計算式変更かを判別できる。モデルの実体（version・パラメータ）は
  **`config` に入れて run と一緒に保存**する（＋`code_commit`）。専用の model ラベルは持たない。
- **予測と評価の分離**: 各 run は予測だけを書き、mAP/AUC のような**サンプル分解できない指標は評価時に集計**。
- **箱とスクリプトの分離**: 箱は配置・命名・不変性・共通メタを固定し、取り込み/評価のロジックは自由。
- **GT の持ち方**: 時系列など構造化された子は schema.yaml の `nested` で**行にインライン**（一覧に出さず
  `ctx.dataset.frames(id)` で取得）。生波形・画像・マスク・点群など重いアセットは台帳に入れず、**scalar 列に
  URL/パスを持たせて評価時に `ctx.read_bytes(url)` で読む**（＝列から派生できる値はデータセットに持たない）。

### ロックレス（S3 だけで排他制御を消す）

CAS もリースも DynamoDB も要らない。4 規則で競合が原理的に起きない:

1. **書き込みは常に新キー**（`rows/{ulid}.json` 等）。上書きしない → write-write 競合なし。
2. **snapshot = 実行時点のキー集合を固定**。run 作成時に LIST し、各ファイルは不変なのでその集合は
   永久に同じ中身を指す（時計ずれ・遅延到着が無関係）。`snapshot_id` は**マージ後の論理内容のハッシュ**
   （生ファイル集合ではない）: 同じ行を再取り込みしても snapshot_id は変わらず（凍結後の再 ingest で
   比較可能性が壊れない）、値の upsert や削除では変わる。内容が同じなら run どうしは比較可能。
3. **導出物は入力名で命名**（`manifest/{ulid}.parquet` + `covered`）。上書きしないので compaction が競合しない。
   読み手は「最新 checkpoint の covered ∪（LIST − covered）」を走査（順序の仮定すら要らない）。
4. **一覧はキャッシュに徹する**。`_index` を真実にせず、LIST で run を並べ、結果は各 `runs/{eval_id}/result/row.json` から直接読む。
   データセットの行は parquet チェックポイント経由だが、これは導出物でいつでも作り直せる。

この規則の外にあるものが 2 つある。**予測はサンプルのキーに書く**ので、同じサンプルに二度書けば後勝ちになる
（1 サンプルは 1 タスクのもので、リトライで答えが 2 つ残る方が困る）。そして **結果は一度だけ書く**（採点済みの
実行への再採点は拒否）が、この確認は原子的ではない。同時に採点すれば両方が書くが、評価器も予測も同じなので
書かれるものも同じになる。

代償はロックでなく**ゴミ回収**に集約される → `cairn vacuum`（古い checkpoint を削除。snapshot が参照する
`rows/` は pin されるので消えない）。

## ライセンス

MIT。[LICENSE](LICENSE) を参照。
