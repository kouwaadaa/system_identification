# System Identification Tool for Tilt rotor UAV
システム同定に必要なツール．Pythonで書いています．

## Description
- 飛行ログの読み込み・閲覧
- 線形システムにおける固有値解析
- データのフィルタリング
- パラメータ推定
- 予測誤差など統計値の算出
- 様々なプロット  
など，データ解析に必要なプログラムが入っています．

## Program Files
- メイン  
`main_tilt0.py` : ティルト角0°の実験データのみ使用  
`main_tilt10.py` : ティルト角10°の実験データのみ使用
- モジュール  
`analyze.py` : 固有値解析など  
`const.py` : 機体情報などの定数ファイル  
`fileread.py` : ファイル読み込み＆計算  
`math_extention.py` : 計算関数群  
`param_estimation.py` : パラメータ推定  
`plot.py` : プロット用（メモ的な側面も）  
`statistics.py` : 統計値の算出  
`thrust.py` : 推力の係数算出

## Usage
### Install
```sh
git clone https://github.com/YubaHiroki/system_identification.git
cd system_identification
```

### Settings
- 必須なライブラリ
  - [NumPy](http://www.numpy.org/)（数値計算）
  - [Pandas](https://pandas.pydata.org/)（データ分析）
  - [Matplotlib](https://matplotlib.org/)（プロット）
  - [sckit-learn](https://scikit-learn.org/stable/)（統計）
  - [IPython](https://ipython.org/)（シェル）
- 必須ではないが，あると便利
  - [Spyder](https://www.spyder-ide.org/)（IDE）
  - [Jupyter Notebook](https://jupyter.org/)（まさにノート，データ分析にも）

### Easy Settings
上記すべてが備わっている[Anaconda](https://www.anaconda.com/)の利用をおすすめします．  
例えば，以下のようなページを参考にインストールしてください．
- for Windows  
  >[Python環境構築 Anacondaインストール](https://techfun.cc/python/anaconda-install.html)
- for Mac  
  >[pyenv + AnacondaでMacにPython環境をセットアップする](https://corgi-lab.com/programming/python/mac-pyenv-anaconda/)
- for Linux  
  >[Ubuntu で Anaconda 3 2018.12 のインストール](https://www.kunihikokaneko.com/dblab/linuxsoft/anaconda.html)

なお，Anaconda環境を構築後，絶対にpipコマンドを使用しないこと．厳重注意．  
condaコマンドで必ずと言っていいほど代用できます．調べましょう．

### Run
- mainの.pyファイルを実行すれば結果が表示されます
- main以外のモジュールを実行した場合は，デバッグに使えます

## Hints

### Developer Environment
- OSとかPythonの環境とか
  - MacOS High Sierra
  - pyenv 1.2.8
  - anaconda3-5.3.0
- 開発環境とか（以下，個人的な偏見が100%）
  - [Atom](https://atom.io/)（テキストエディタ，今すぐEmacsなんてやめなさい）
  - Spyder（実行結果からPandasの表が見れるのが良い）
  - Excel,Googleスプレッドシート（ログがCSVファイルなので）
  - [zsh,prezto](https://qiita.com/s_s_satoc/items/e3c1b9b3545fd572dd1c)（君のターミナルはださい）
  - [Markdown](https://ja.wikipedia.org/wiki/Markdown)（進捗報告，引き継ぎ，などなんでも）

### Study References
- UAVやドローンについて知る
  - [浦久保孝光 : “VTOL型ドローンの研究開発―次世代ドローンの実現に向けて”,  一般社団法人システム制御情報学会 (2016)](https://www.jstage.jst.go.jp/article/isciesci/60/10/60_437/_pdf)
  - ドローンとかUAVってYoutubeで調べる
- 航空機力学を学ぶ
  - [鳩ぽっぽ 初心者のための航空力学講座](https://pigeon-poppo.com/)
  - 加藤寛一郎, 他 : “航空機力学入門”, 東京大学出版会 (1982)
  - 片柳亮二 : “航空機の飛行力学と制御”, 森北出版株式会社 (2007)
  - 嶋田有三, 他 : “飛行力学”, 森北出版株式会社 (2017)
  - Bernard Etkin et al. : “Dynamics of FLIGHT”, John Wiley Sons. Inc. (1959)
  - Eugene A.Morelli et al. : “Aircraft System Identification Theory and Practice Second Edition”, Sunflyte Enterprises (2016)
- システム同定について学ぶ
    - 成岡優 : “システム同定による小型無人航空機の飛行特性の取得”, 東京大学大 学院工学系研究科博士論文 (2016)
- 諸先輩方の論文

### Develop References
とにかく，基本的には公式のリファレンスを参考にすることをおすすめします．
- Python関連
  - [note.nkmk.me](https://note.nkmk.me/)（Pythonの記法に困ったらここ）
  - [PEP8](https://pep8-ja.readthedocs.io/ja/latest/)（コーディング規約，几帳面な方向け）
  - [Pythonで可視化入門](https://pythonoum.wordpress.com/2019/01/18/python%E3%81%A7%E5%8F%AF%E8%A6%96%E5%8C%96%E5%85%A5%E9%96%80/)（色々とツールがあります）
- Pixhawk
  - [PX4](https://px4.io/)（公式）

## Contribution
複数人で共同管理する場合は以下の手順で．
1. Fork it  
2. Create your feature branch  
3. Commit your changes  
4. Push to the branch  
5. Create new Pull Request

## Contact
どうしても，どうしても，困ったときは  
<&#121;&#117;&#98;&#97;&#51;&#104;&#105;&#114;&#111;&#107;&#105;&#50;&#54;&#64;&#103;&#109;&#97;&#105;&#108;&#46;&#99;&#111;&#109;>  
まで，お気軽にどうぞ．
