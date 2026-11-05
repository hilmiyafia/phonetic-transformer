# Phonetic Transformer
This repository contains the code of a transformer to convert English text to its phonetic transcription.

## Demo
Try the live transformer now!<br>https://hilmiyafia.github.io/phonetic/

## Training
Before training, you can configure the parameters through `config.py`, then run:<br>
~~~
python train.py
~~~

## Inferring
~~~
python infer.py "this is a test"

python infer.py [text to convert]
~~~

## Export to ONNX
~~~
python export.py
~~~
