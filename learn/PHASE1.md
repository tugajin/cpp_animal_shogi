# フェーズ1 手順書（PO サーモメーター: エンジン側の測定）

問い: 探索予算 po_num を上げると、教師 s と探索の指し手は実際に良くなるのか。
あわせて、詰み探索が生成時にどれだけ効いているか（発火率）を測る。
ツイン診断は Python で完結するので PHASE0.md 側に入れてある。

## 方法A（近似版・推奨の第一歩）: po 違いの小規模生成を2本作って比べる

コード変更は1行だけ: `ai/selfplay.hpp` の `DescentSearcherLocal::selfplay()` 内にある
`this->po_num = 100;` を、生成ごとに書き換える（100 のままの版と 400 の版）。

手順:
1. 現在の `learn/data/` を退避する（`mv data data_backup_exp4` など）
2. po=100 のまま 2,000 局生成し、出力を `data/gen_po100/` に移す
3. po_num を 400 に書き換えて再ビルドし、2,000 局生成して `data/gen_po400/` に移す
4. po_num を 100 に戻す
5. 比較（learn/ で）:
   ```bash
   ../.venv/bin/python investigate45.py --group gen_po100 --out phase1_po100.md
   ../.venv/bin/python investigate45.py --group gen_po400 --out phase1_po400.md
   ../.venv/bin/python audit_order.py --group gen_po400 --out phase1_audit_po400.md
   ```
   - investigate45 の C（教師の正解率、カテゴリ別）と B の |s|<0.2 率を po 間で比べる
   - audit_order の C（実着手のカテゴリ維持）を、既存の po=100 の audit 結果と比べる

判定:
- C の教師正解率・実着手維持率が po=400 で明確に上がる → 探索予算は生成レバーになる（段階B は B2 予算ミックスが有力）
- ほぼ変わらない → 葉（モデル）の質が限界で、PO 系のレバーは全滅。学習側 arm（段階A）に集中する

注意: 2,000 局同士の比較なので、C のサンプル数（既定 20,000 件上限）で見ること。
生成条件（平手/ランダム開始の比率、ε）は変えない。

## 方法B（正確版・任意）: 詰み探索の発火率を数える

`ai/ubfm.hpp` のノード評価には mate 判定のブロック（`#if 1` で囲まれた
`mate::mate_search(pos,5)` / `mate::mated_search(pos,4)` の箇所）が2か所ある。
そこにカウンタを足す:

- 評価総数、mate_search で NodeWin になった回数、mated_search で NodeLose になった回数を
  グローバルな atomic カウンタで数え、自己対局終了時に割合を表示する
- 発火率が低い（数%以下）なら、詰み探索は現行の深さでは教師にほぼ寄与しておらず、
  「深い詰みで解ける失敗が多いか」（audit の失敗局面への深い mate_search のオフライン適用）と
  合わせて B3（詰み深化）の判断材料になる
- 発火率が高いのに教師が改善しないなら、律速は詰みではなく葉の評価

※ ローカル側のコードはクローンと差分がある可能性があるため、カウンタの追加は
「mate 判定で resolved になった分岐」に足す、という置き場所の指定に留める。
具体的な差分が必要なら、対象箇所の現行コードを貼ってもらえればこちらで書く。

## 返してほしいもの

- 方法A: phase1_po100.md / phase1_po400.md / phase1_audit_po400.md の内容
- 方法B（実施した場合）: 発火率の表示結果
