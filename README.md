# FV_judgement

判断ロジックのまとめ：

・判断の前に基準にとして女性向け・男性向け言葉のリストを作られました。
・代表的な女性向け・男性向けFVを選んで、Google API による色トーンの情報を抽出されます。

FVの判断流れ：

FVのテキスを取り出して、テキストの中で女性向けの言葉あれば、色情報を代表的な女性向けFVの色情報と比べます。
Normalized Euclidean Distance　＜ 0.5 ≒> result: For woman
Normalized Euclidean Distance　> 0.5 ≒> result: For man

FVのテキスを取り出して、テキストの中で男性向けの言葉あれば、色情報を代表的な男性向けFVの色情報と比べます。
Normalized Euclidean Distance　＜ 0.5 ≒> result: For man
Normalized Euclidean Distance　> 0.5 ≒> result: For woman

上の以外の場合は結果は Elseになります。





