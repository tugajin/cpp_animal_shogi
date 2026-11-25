# CN 選択の生成実験 手順書（CN_EXPERIMENT.md）

問い:
1. **生成比較**: 同じ po=100 で、葉選択を共謀数（CN）基準に変えると、教師 s は UBFM 選択より正確になるか（gen_po100 との比較、サーモメーター方式）
2. **注釈の妥当性**: レコードに付く CN は「その探索結論の信頼度」として機能するか（CN が大きい局面ほど s が当たるか / CN は維持手数を木内部から再現できるか）

## 前提条件

- **モデルは gen_po100 生成時と同じもの**（best_single.h5 が変わっているなら、比較が成立しないので UBFM 側も再生成すること）
- po_num=100、2,000 局、生成条件（平手/ランダム開始 50/50、ε=0.1）は gen_po100 と同一
- 変わるのは葉選択だけ。バックアップ（update_node）、予算、詰み探索、interrupt_descent の早期終了条件は両 arm で共通

## コード変更（cn_search.patch）

`ai/ubfm.hpp`:
- `Node` に `cn_cost[4]` を追加（[0]=GE(+0.2) [1]=LE(+0.2) [2]=GE(−0.2) [3]=LE(−0.2)。GE(τ)=値を τ より大きくするのに必要な葉評価の変更数、LE(τ)=その逆）
- `compute_cn()` / `root_category_cn()`: 閾値 ±0.2 に対する共謀数の木全体への再帰計算。葉（未展開・未解決）はコスト 1、解決済みノードは CN_INF（葉の変更では動かない）
- `select_leaf_cn()`: 根のカテゴリを変える最小共謀を辿って葉を1つ選ぶ。辿れない場合は `leaf == nullptr` を返し、呼び出し側が通常の VL 選択にフォールバックする
- フラグ `g_use_cn_search`（既定 false）

`ai/selfplay.hpp`:
- `search_descent()`: フラグ ON のとき、バッチ前に `compute_cn()` し直して CN 選択を使う。根の CN が CN_INF（証明済み級）の間は共謀が存在しないため通常選択に任せる
- `execute_descent()`: フラグ ON のとき、最終的な木から根 CN を計算してレコードに `"c"` を注釈する（CN_INF は −1 で記録）。既存の {p, s, r} は無変更で、Python 側の既存ローダは余分なキーを無視する

検証済みのこと（こちらで実施）:
- 再帰の正しさ: Python でランダムな minimax 木 400 本を作り、葉の部分集合の全探索＋区間伝播によるブルートフォースと完全一致。降下規則が最小共謀の葉に着くことも 328 件全件で確認
- C++ コア（compute_cn / root_category_cn）は ubfm.hpp から抽出して単体コンパイルし、手計算ケース（CN=1 / 2 / ∞ / draw 根）で一致を確認
- プロジェクト全体のビルドはそちらの環境でのみ可能なので未実施

既知の近似:
- バッチ内では cn_cost はバッチ開始時点の値を使う（バッチ内の展開は次のバッチの compute_cn で反映）。VL の vl カウンタと同じ性質の割り切り
- 生成時間は compute_cn の分だけ増える。木は数百ノード規模なので誤差範囲のはずだが、極端に遅い場合は知らせてほしい

## 手順

1. パッチを当てる（リポジトリ直下で）:
   ```bash
   git apply learn/cn_search.patch
   ```
   ローカル側のコードとの差分で当たらない場合は、無理に当てず、対象箇所（ubfm.hpp の Node / select_leaf_vl 周辺、selfplay.hpp の search_descent / execute_descent / ReplayBuffer）の現行コードを貼ってください。差分を書き直します
2. `ai/ubfm.hpp` の `g_use_cn_search` を `true` に書き換える
3. ビルドして 2,000 局生成する（gen_po100 と同じ方法）。出力 JSON を `learn/data/gen_cn100/` に移す
4. `g_use_cn_search` を `false` に戻す
5. 分析（learn/ で）:
   ```bash
   ../.venv/bin/python analyze_cn.py
   ```
   （既定: --cn-group gen_cn100 --base-group gen_po100、出力 analyze_cn_result.md）

## 判定

- 問い1で CN 側の win/lose 正解率が明確に高い（+2pt 以上）→ 葉選択の CN 化は生成レバーになる。段階B の候補として B1（木学習）と比較する
- 問い1が互角でも、問い2で CN 層別の勾配（CN が大きいほど s が当たる）があれば、CN は注釈（信頼度）として採用できる
- 問い1で CN 側が低く、問い2にも勾配が無ければ、この路線は下げる
- 問い3（c=−1 は格子一致か）が崩れていたら、C++ 実装側の不整合を先に疑う

## 返してほしいもの

- analyze_cn_result.md の内容
- 生成にかかった時間の感触（現行比で極端に遅くないか）
- パッチが当たらなかった場合は、その箇所の現行コード
