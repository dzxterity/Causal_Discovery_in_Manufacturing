# Neural relational inference for interacting systems

Оригинальные репозиторий и статья доступны по ссылкам:
https://github.com/ethanfetaya/NRI

**Neural relational inference for interacting systems.**  
Thomas Kipf*, Ethan Fetaya*, Kuan-Chieh Wang, Max Welling, Richard Zemel.  
https://arxiv.org/abs/1802.04687  (*: equal contribution)

В данной работе мы адаптировали данные под данные химического производства, взятые из маленького датасета.

Для использования необходимо запустить train-my.py, предварительно подгрузив данные. Он обучит энкодер. Сделать это можно в run.ipynb.
После этого необходимо запустить энкодер на наших данных и усреднить полученную матрицу в файле We_create_graph.ipynb.
У вас получится граф, который можно будет протестировать в файле test_your_causal_discovery_method-v2.ipynb или test_on_BIG_TEP.ipynb.
Надо отметить, что на больших данных полученный граф дает результаты хуже, чем единичная матрица и матрица корреляции с 200 ребер.


### Cite
Я копирую это с оригинального репозитория:
```
@article{kipf2018neural,
  title={Neural Relational Inference for Interacting Systems},
  author={Kipf, Thomas and Fetaya, Ethan and Wang, Kuan-Chieh and Welling, Max and Zemel, Richard},
  journal={arXiv preprint arXiv:1802.04687},
  year={2018}
}
```
