# 予算の蒸留（実験候補 A）ランブック

## 問いと対照

- 問い: 学習対象を「モデルの素の値（0回探索）」から「浅い探索（po=10）の
  出力」に変え、深い探索（po=100）の結果に近づけると何が変わるか。
- 対照は **arm 0**（同じ生成・同じ root 局面・同じ累計予算 100k 局面で、
  素の値を教師 s に合わせる通常学習）。生成条件を揃えるため、この実験の
  生成は arm 0 と同一（USE_DESCENT=false、CN オフ、po=100、ε=0.1）。
- 判定: probe の P/R を po=10 と po=100 の両方で測る（budget_curve.py）。
  蒸留が効いていれば、学習が進むほど「po=10 の出力が po=100 に近づく」
  （固定点の署名）。併せて audit A/B（対 arm 0）と ans（参考）。

## 1. ビルド

差し替えるファイルは `ai/main.cpp` と `ai/selfplay.hpp`（dump モード追加。
probe の po 引数化を含む版）。ビルド条件は arm 0 と同じ:

- `ubfm::g_use_cn_search` は触らない（実行時フラグ。生成は cn_search=0 で回す）
- `USE_DESCENT=false`（selfplay.hpp の static constexpr、そのまま）
- バイナリ名はスクリプト側で `./cpp_animal_shogi_distill` を想定。
  ローカルの流儀（サフィックスなし）に合わせて変えてよい

## 2. 初期化（1回だけ）

```
cd learn
../.venv/bin/python train_distill.py --model-dir model/distill --init-only
```

seed=123 の乱数初期モデルが model/distill/ に出る（全 arm と共通）。

## 3. 関門: 最初の dump と backup 検証（初回のみ）

ループを回す前に、dump と backup 再現性を1回だけ手で確認する:

```
cp model/distill/best_single.onnx model/best_single.onnx
./cpp_animal_shogi_distill 250 > logs/distill_check.log 2>&1   # 生成（PARAMS 確認）
mkdir -p data/distill_gen0
mv data/selfplay*.json data/distill_gen0/   # ※この世代を gen0 として使う場合
../.venv/bin/python train_distill.py --model-dir model/distill \
    --data-dir data/distill_gen0 --make-keys --keys-out data/distill_gen0/keys.txt
./cpp_animal_shogi_distill dump data/distill_gen0/keys.txt 10 data/distill_gen0/trees.txt
../.venv/bin/python train_distill.py --model-dir model/distill \
    --trees data/distill_gen0/trees.txt --verify
```

`verify: OK`（全ての木で、ダンプした葉の値を入れると root_w が再現される）
が出れば backup の実装が C++ の探索と一致している。以後のイテレーション
でも train_distill.py は学習前に同じ検証を自動で回し、再現できない木が
あれば警告する。

## 4. ループ実行

```
sh run_distill.sh 250 20 5000 10
```

引数は games / 総イテレーション数 / 1回の学習局面数 / 学生 po。
run_arm.sh と同じ流儀で、arm_state.json から再開し、PARAMS と
DUMP_PARAMS を毎回確認して食い違えば止まる。

手作業で1イテレーション回す場合は §3 の生成→make-keys→dump の後に:

```
../.venv/bin/python train_distill.py --model-dir model/distill \
    --data-dir data/distill_gen0 --trees data/distill_gen0/trees.txt \
    --init model/distill/best_single.h5 --iter-positions 5000 --student-po 10
```

## 5. 評価

- ans は train_distill.py が毎回 train_log.md に記録する（参考指標）
- 固定点の読み: 学習済みモデルを差し替えて probe を po=10 / po=100 で回し、
  budget_curve.py で arm 0 の曲線と比較する。po=10 側の P/R が po=100 に
  どれだけ迫ったかを見る
- audit A/B: audit_order.py で arm 0 と同じデータグループに対して測る

## 6. 注意

- 損失の値は train_arm.py とは別物（木ごとの backup 誤差）なので横比較
  しない。ログの `mean|backup-s|` が蒸留の進み具合の目安
- backup は Python 側で再帰するため、epoch 時間は train_arm より長く
  なりうる。遅すぎる場合は --batch-size を下げるより先に、まず1回の
  所要時間を教えてほしい（高速化の余地はある）
- `--backup soft --tau 0.1` で soft backup 変種になる（勾配が全葉に
  分散）。本線は hard で回し、soft は変種として分けること
- 学生 po を変える場合は dump の po も変える（木はイテレーション毎に
  作り直すので、途中変更はその回から効く。train_log の記録と揃えること）
- 教師 s は生成時のモデルによる po=100 の値で、イテレーション毎に
  更新される（現行ループと同じ自己蒸留構造）
