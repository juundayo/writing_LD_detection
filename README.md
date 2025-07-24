### Description
An AI OCR model trained on greek letter recognition using PyTorch.
The final code will take written text as an input and then recognize if
the person who wrote it has potential writing and/or spelling disorders.


------------

### Handwritten Greek text OCR model: 
The OCR model is trained on two custom datasets.
For the first one, we asked ~100 volunteers to fill out a
form of single characters that contained every letter of
the Greek alphabet, as well as every tonos and dialytic 
variation.
For the second one, the national Centre for Scientific Research
Demokritos offered us the dataset used for the ICDAR2012 Writer
challenge. It includes 2 passages (~35 words each) that 100 
different writers wrote, and we used it to boost our initial dataset.

