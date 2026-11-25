# 実験5 4アーム factorial 実行手順（ARM_RUNBOOK.md）

設計は EXP5_DESIGN.md の「実施計画（2026-11-21 改訂）」。学習側は train_arm.py。
全 arm 共通の条件: 乱数初期化 SingleNet（seed=123 で全 arm 同一の初期重み）、
生成→学習を約20イテレーション、生成は累計 5,000局・学習局面は累計 100k 上限
（上限の管理は train_arm.py が model/arm{N}/arm_state.json で行う）。
揃えるのは総予算の上限とイテレーション回数であり、1イテレーションの局数は
揃えない。各 arm の records/game（§5 で実測）から
「1回の局数 ≒ 100,000 ÷（records/game × 20）」で決める。根のみ arm では
records/game ≒ 21（≒平均手数）なので約250局になる。B1 系は records/game が
桁違いに大きいため、1回の局数はずっと少なくなる。

## arm とフラグの対応

フラグは2つともコンパイル時切替なので、arm ごとにバイナリを分ける。

| arm | g_use_cn_search（ai/ubfm.hpp:21） | USE_DESCENT（ai/selfplay.hpp:23） | 記録 |
|---|---|---|---|
| 0 Baseline | false | false | 根局面のみ |
| 1 B5 | **true** | false | 根局面のみ（c 注釈つき） |
| 2 B1 | false | **true** | 木学習 |
| 3 B5+B1 | **true** | **true** | 木学習（c 注釈つき） |

## 1. arm ごとのビルド（各 arm の初回だけ）

```bash
cd ~/Documents/cpp_animal_shogi
# 例: arm 1（CN 選択）。arm ごとに true/false の組を変える
sed -i 's/inline bool g_use_cn_search = false;/inline bool g_use_cn_search = true;/' ai/ubfm.hpp
sed -i 's/static constexpr bool USE_DESCENT = false;/static constexpr bool USE_DESCENT = true;/' ai/selfplay.hpp  # arm 2, 3 のみ
cmake --build ai/build -j
cp ai/build/cpp_animal_shogi learn/cpp_animal_shogi_arm1
# フラグを戻してから次の arm をビルドする（git diff で状態を必ず確認）
```

ビルド自体は普段の手順のままでよい。要点は上記2フラグの組だけ。
ビルド後に `grep -n "g_use_cn_search =" ai/ubfm.hpp` と
`grep -n "USE_DESCENT =" ai/selfplay.hpp` で、意図した組になっているか確認すること。

## 2. arm の初期化（各 arm の初回だけ）

```bash
cd ~/Documents/cpp_animal_shogi/learn
../.venv/bin/python train_arm.py --arm 1 --model-dir model/arm1 --init-only --seed 123
```

seed が同じなら全 arm で同一の初期重みになる（train_arm.py は seed を固定してから
SingleNet を構築する）。model/arm1/ に best_single.h5 / .onnx / .jit.pt が出る。

## 3. イテレーション（1 arm 分、20回。1回の局数 GAMES は arm ごと）

データの置き場所は既存ツール（audit_order の --group 等）と衝突しないよう
フラットな名前 `data/arm{N}_gen{i}/` にする。ループは run_arm.sh にまとめた。

```bash
cd ~/Documents/cpp_animal_shogi/learn
./run_arm.sh 1 250   # ./run_arm.sh <arm> <games> [total=20]
```

run_arm.sh が各イテレーションでやること:

1. arm の ONNX を model/best_single.onnx に差してから生成する
   （生成側バイナリは固定パスしか読まないため、この差し替えが必須。
   忘れると前の arm のモデルで生成してしまう）
2. 生成ログ（logs/arm{N}_gen{i}.log）の PARAMS 行で use_descent /
   cn_search を検証し、その arm の期待値と違えばバイナリの取り違えとして止める
3. 生成ファイルを data/arm{N}_gen{i}/ に移す
4. train_arm.py で学習する。失敗したらループを止める
   （古いモデルのまま生成を続けないため）

- 中断しても同じコマンドで再実行すればよい。学習の完了数は arm_state.json の
  iterations で管理され、そこから再開する。学習中に落ちた回は生成済みデータを
  再利用して学習からやり直す
- 予算は上限であって割り当てではない。train_arm.py が累計予算
  （5,000局 / 100k 局面）を arm_state.json で管理し、超える分は対局単位で
  間引く（train_log.md に記録が残る）。B1 系では局面数上限が先に当たり、
  生成した対局の総数は 5,000 に届かないのが正常
- 各イテレーションの学習後、train_arm.py が ans（全体と win/lose のみ）を
  model/arm{N}/train_log.md に自動で記録する

## 4. 評価

```bash
# 世代ごとの監査（例: arm 1 の最終世代）
../.venv/bin/python audit_order.py --model model/arm1/best_single.h5 \
    --group arm1_gen_19 --out audit_arm1_gen19.md
```

判定は audit の win 維持率と、train_log.md の効率曲線（イテレーションごとの
ans 推移）の arm 間比較。draw の数値は参考値。

## 5. 初回生成の健全性チェック（本番の前に各 arm 10局だけ回す）

- arm 1, 3（CN）: レコードに "c" 注釈が付いていること
- arm 2, 3（木学習）: 1ファイルのレコード数が手数より大幅に多いこと
  （根のみ記録なら ≒ 手数）。想定と違えばフラグの組を疑う
- arm 0〜3 共通: 生成ファイル数が指定局数と一致すること
- 生成ログ先頭の `PARAMS` 行（search_params_log.patch で追加）で、その arm の
  フラグ組（use_descent / cn_search）と po_num が意図通りか確認すること。
  バイナリの取り違えはここで検出できる
- **records/game の実測**: 10局分のレコード総数 ÷ 10 を records/game とし、
  1イテレーションの局数を GAMES ≒ 100,000 ÷（records/game × 20）で決めて
  §3 のループに設定する。根のみ arm では records/game ≒ 21 なので
  GAMES ≒ 250 になる。ここを実測せずに 250 に固定すると、records/game の
  大きい arm では局面予算の大半を未学習モデルのデータで食い尽くして
  しまうため、必ず実測から決めること
- **B1 系の実測値と設定（2026-11-21 arm 2 で実測、同日改訂）**: records/game
  ≒ 2,000（≒ 100/手 × 平均手数）だが、木レコードは生成時にほぼ無償で
  増えるため、生成局数は根 arm と揃える（GAMES=250）。学習局面だけを
  train_arm.py の --iter-positions で 1イテレーション 5,000 局面に絞り、
  全対局の木レコードからレコード単位で一様にサンプルする
  （`sh run_arm.sh 2 250 20 5000`）。対局単位で間引くと学習局面が
  数十局分に偏って多様性を失うので、その方式は採らない

## 6. 既知の制約

- **木学習 arm の引き分け対局は学習側で除外できない**: descent のレコードには
  対局結果が残らない（r=-99）ため、train_arm.py では判定不能として全て残し、
  ログにその旨を出す。実験4では引き分け対局はレコードの 1.8% で影響は小さい
  見込みだが、気になる場合は生成側（C++）で引き分け対局のファイルを
  書き分ける変更が別途要る。
- 木学習 arm は1局あたりのレコード数が大きい。100k 局面の上限に先に当たる
  のはこの arm で、間引きの結果として学習に使う対局数が他 arm より少なくなる
  のが正常（「同じ局面数で比べる」が予算の定義）。
- train_arm.py はオプティマイザを引き継がない（毎回新規）。重みだけを
  --init で引き継ぐのが実験5の取り決め。

## 7. 段階A: A9L（兄弟ランキング学習）の実行

生成は arm 2 と同一（バイナリは arm 2 の設定のものをそのまま使う）。学習だけが train_rank.py に変わる。

```bash
cd ~/Documents/cpp_animal_shogi/learn
# 初期化（factorial と同じ seed 123 の乱数初期モデル）
../.venv/bin/python train_rank.py --model-dir model/a9l --init-only --seed 123
# 事前確認: arm 2 の既存データで兄弟グループがどれだけ再構成できるか（学習しない）
../.venv/bin/python train_rank.py --analyze --data-dir data/arm2_gen_19
# 本番（生成→学習を20イテレーション）
sh run_rank.sh 250 20 5000
# 監査（対照は arm 2 の最終モデル。同じ2グループで測る）
python3 audit_order.py --model model/a9l/best_single.h5 --group arm0_gen_19 --out audit_a9l_on_arm0gen.md
python3 audit_order.py --model model/a9l/best_single.h5 --group arm1_gen_19 --out audit_a9l_on_arm1gen.md
```

- run_rank.sh の冒頭に `run_rank.sh version 2026-11-22 v1` が出ること。train_rank.py の開始時にも version 行と self-test: OK が出る
- train_log.md（model/a9l/）に各回の「グループ再構成」の行が出る: 走査した親の数・グループ数・子数の平均。グループが極端に少ない回は順序損失が効いていないので、その場合は --analyze の結果と合わせて知らせること
- 学習局面数の会計は arm 2 と同じ（5,000/回・累計 100,000）。グループで不足した分は一様サンプルのレコード（MSE のみ）で補充され、ログに内訳が出る

## 8. B1 対照実験（SEED / RAW / DEPTH）の実行

設計と読みは EXPERIMENT.md「B1 対照実験の設計」。共通: 土台は arm 2 レシピ、
生成 po10、run_arm.sh v6（SEED / EXTRA_TRAIN_ARGS 対応）、train_arm.py v7。
対照は arm2po10（seed 123）。

```sh
# SEED arm（ノイズフロア）: 道具は run_arm.sh v6 のみ。バイナリは arm2po10 と同じ
../.venv/bin/python train_arm.py --arm 2 --model-dir model/arm2po10s456 --init-only --seed 456
SEED=456 MODEL_DIR=model/arm2po10s456 sh run_arm.sh 2 250 20 5000 10

# RAW arm（ラベルの形）: バイナリは通常の USE_DESCENT=true ビルドでよい
../.venv/bin/python train_arm.py --arm 2 --model-dir model/arm2raw --init-only --seed 123
EXTRA_TRAIN_ARGS="--relabel-raw" MODEL_DIR=model/arm2raw sh run_arm.sh 2 250 20 5000 10

# DEPTH arm（深さ <=1）: 深さ記録対応の selfplay.hpp でビルドしたバイナリが必須
# （レコードに "d" が無いと train_arm.py が初回で中止するので、取り違えはそこで止まる）
../.venv/bin/python train_arm.py --arm 2 --model-dir model/arm2d1 --init-only --seed 123
EXTRA_TRAIN_ARGS="--max-depth 1" MODEL_DIR=model/arm2d1 sh run_arm.sh 2 250 20 5000 10

# 監査（全 arm 共通: 両グループで A/B を測る）
../.venv/bin/python audit_order.py --model model/<arm>/best_single.h5 --group arm0_gen_19 --out audit_<arm>_arm0gen.md
../.venv/bin/python audit_order.py --model model/<arm>/best_single.h5 --group arm1_gen_19 --out audit_<arm>_arm1gen.md
```

- train_arm.py の開始時に `train_arm.py version 2026-11-24 v7` と self-test: OK が出ること
- RAW arm は train_log.md に「ラベル付け替え」の行（付け替えた件数と |新 s − 旧 s| の平均）が出る
- DEPTH arm は train_log.md に「深さフィルタ」の行（残った局面数）が出る。po10 の木では d<=1 は母集団の約2割の見込みで、5,000 件のサンプルには十分足りる
