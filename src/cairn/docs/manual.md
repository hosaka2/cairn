# 使い方

cairn は**評価とデータセットの台帳**です。DB を立てず、オブジェクトストレージ（S3 / GCS / ローカル）
だけで、追記されるデータセットと積み上がる評価結果を記録・比較します。

学習やモデル管理（MLflow / W&B）、LLM トレース（Langfuse）、推論の実行（Dagster 等）はスコープ外です。

## 1. 置き場所を決める

すべては `CAIRN_ROOT` の下に置かれます。解決順は **`--root` > 環境変数 > カレントの `.env`**。

```bash
cairn --root file:///tmp/cairn-demo dataset ls     # フラグ
export CAIRN_ROOT=s3://bucket/cairn                 # 環境変数
```
```env
# .env（カレントに置くと自動で読まれる）
CAIRN_ROOT=s3://bucket/cairn
AWS_PROFILE=myprofile        # 認証は環境変数 / ~/.aws / IAM ロールを自動で拾う
AWS_ENDPOINT_URL=http://localhost:9000   # MinIO を使う場合だけ
CAIRN_LANG=ja                            # マニュアルとフォーム説明を日本語に（未設定なら英語）
CAIRN_TRACEBACK=1                        # エラーをメッセージでなくトレースバックで出す（スクリプトのデバッグ用）
```

## 2. まず動かす

```bash
cairn demo-seed     # 合成データで全機能を実際に動かす（faked なし）
cairn web           # http://127.0.0.1:8000
```

## 3. 3 つの登場人物

| | 役割 | 単位 |
|---|---|---|
| **データセット** | 評価の対象。**気軽に追記**して充実させる台帳 | 1 行 = 1 サンプル |
| **評価テーブル** | 「何を・どの指標で測るか」の定義。一覧の列を決める | 1 テーブル = 1 種類の評価 |
| **実行（run）** | 1 回の測定。予測を書き、評価して結果を残す | 1 実行 = 石ひとつ |

評価一覧では**実行が石として積み上がります**（石の幅＝主要指標）。

## 4. 画面から一周する

1. **データセットを作成** — `schema.yaml`（列の定義）を貼る。`key` が行の identity。
2. **データを追加** — 取り込みスクリプトのフォーム、または JSONL の貼り付け。
   プレビューで確認 → 確定で初めて追記されます。同じ `key` の再追記は上書き（後勝ち）。
3. **評価テーブルを作成** — `table.yaml`（一覧に出す列）を貼る。対象データセットは選択できます。
4. **新しい実行** — タイトル（必須）・データセット・評価方法・設定を決めて発行する。
   cairn は推論をしません。予測が書き込まれるまで実行は **Running** で待ちます（5 を参照）。
   demo のスクリプトなら `cairn eval run <table> <eval_id>` が書きます。
5. **Evaluate** — 予測が揃ったら待機中の実行で押す。結果が石として積まれます。
6. **比べる** — 実行を選び「別実行と比べる」。結果表に行が増え、差が色付きで出ます
   （青 = 良い方向 / 赤 = 悪い方向。向きは列ごとの `direction`）。

## 5. オーケストレータから回す

cairn は推論をしません。推論は動かす場所——オーケストレータ（Dagster / Airflow / バッチキュー）でも
自前のスクリプトでも——で起き、cairn が持つのはその両端だけです:

```bash
# 1. 実行を作る。ここでデータが凍る（web の「Run」も同じ）
EID=$(cairn eval create-run anomaly --dataset sensor-anomaly-A --evaluator-version v1 \
        --title "nightly batch" --config '{"weights": "s3://models/2026-08-26.pt"}')

# 2. 外側のタスクごとに: 何をやるか / 結果をどこに書くか
cairn eval targets anomaly $EID                  # サンプル id を 1 行 1 件で
cairn eval put-prediction anomaly $EID --sample-id A_007 --file pred.json --ext json

# 3. 全タスクが終わったら、まとめて採点する
cairn eval score anomaly $EID --evaluator evals.anomaly.v1:EVALUATOR
```

発行した時点で評価画面の **Running** に出ます。その版の評価器が登録されていれば **Evaluate** も出ます。
ジョブがどこまで進んだかはオーケストレータ側の話で、台帳は「発行済みで結果がまだ無い」ことだけを持ちます。
snapshot は 1 の時点で凍るので、ジョブが走っている間にデータセットへ追記しても、この実行の測定対象は変わりません。

予測は不透明なバイト列で、読むのはそのテーブルの評価器だけです。`--jsonl` は
`{"sample_id": …, "prediction": …}` の行をまとめて書きます。どちらの形も、採点済みの実行には
書き込みません（結果は、書かれた時点の予測の上に立っているため）。

## 6. 結果が比較できる条件

cairn は結果を **2 軸**で刻みます。

- **`snapshot_id`** — 実行時点のデータセットの内容（中身のハッシュ）。**一致する実行どうしだけ比較できます**。
  同じ行を再取り込みしても変わりませんが、値を直したり行を消すと変わります。
  画面では、データが違う実行も見たいときのために並べられますが、その行には `other data` と印が付きます。
- **`evaluator_version`** — 評価式の版（`v1`, `v2`…）。式を変えたら必ず版を切ること。
  「先月 0.81 / 今月 0.79」がデータの変化か計算式の変更かを、これで判別できます。

モデルの識別（重みの版・ハイパーパラメータ）は専用の欄を持たず、**実行の `config`** に入れて
`code_commit` と一緒に保存します。

## 7. CLI

```bash
cairn check                     # スクリプトが契約どおりか検査（書いたらまずこれ）
cairn init                      # datasets/ evals/ を作る
cairn new dataset <name>        # schema.yaml + ingest.py の雛形
cairn new eval <name>           # table.yaml + run.py + v1.py の雛形
cairn new eval-version <name> v2  # 評価式の新版（旧版は編集しない）

cairn dataset ls
cairn dataset create --schema schema.yaml
cairn dataset ingest <name> --jsonl rows.jsonl   # 大量データはこちら（web の貼付欄は少量向け）
cairn dataset show <name>
cairn dataset delete <name> --key a --key b        # tombstone。snapshot が変わる

cairn eval ls
cairn eval create-table --table table.yaml
cairn eval create-run <table> --dataset <ds> --evaluator-version v1 \
    --title "何の実行か" --config '{"threshold": 0.5}'
cairn eval targets <table> <eval_id>               # この実行が対象にしているサンプル id
cairn eval put-prediction <table> <eval_id> --sample-id a --file pred.json --ext json
cairn eval put-prediction <table> <eval_id> --jsonl preds.jsonl   # まとめて書く（1 行 1 予測）
cairn eval run <table> <eval_id>                   # 登録済みスクリプトでこの場で推論する
cairn eval score <table> <eval_id> --evaluator my.module:MyEvaluator
cairn eval withdraw <table> <eval_id> --reason "重みが違った"
                                # 実行を一覧から取り下げる（取り消し不可。記録は残る）
cairn eval rescore <table> <eval_id> --evaluator my.module:V2 --evaluator-version v2
                                # 同じ予測を別の採点方法で（新しい実行として残る）
cairn eval show <table> <eval_id>

cairn vacuum                    # 古い checkpoint を回収（導出物のゴミ。データは消えない）
cairn docs scripting            # スクリプトの書き方
```

## 8. web をどこで動かすか

`127.0.0.1` で待ち受け、ログインは無い。到達できる人は誰でもデータを追加し、実行を作り、採点できる。
localhost に留めるか、認証するプロキシの背後に置くこと（`auth_header`＝`X-Forwarded-User` は
そのための設定で、その名前が `created_by` になる）。

## 9. 仕組み（知っておくと迷わない）

- 書き込みは**常に新しいファイル**として積まれ、上書きしません。だからロックが要りません。
- データセットの現在状態は、追記ログを順に適用した結果です（同じ `key` は後勝ち、削除は tombstone）。
- データセットの行一覧は parquet チェックポイント経由ですが、**真実は JSON 側**でいつでも再構築できます。
  評価結果の方は JSON をそのまま読みます（run 1 件 = ファイル 1 個）。
- `cairn vacuum` はキャッシュ（古い checkpoint）だけを消します。実行が参照するデータは消えません。
