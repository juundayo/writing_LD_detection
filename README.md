## Greek Language Writing and Spelling Disorder Detection 
An application that takes handwritten text as input, and produces a writing and spelling disorder classification based on image processing tools, machine learning and a custom-made OCR model. Developed as part of my graduation thesis.

<img width="1920" height="1080" alt="pipeline" src="https://github.com/user-attachments/assets/28e3b63d-f9d6-4191-985e-0c0f1011a22d" />

## Handwritten Greek text OCR model 
A Res-Net-style architecture, with some upgrades tailored for handwritten single-character recognition that include Squeeze-and-Excitation blocks (SEBlocks), Multi-Head Attention layers, and the addition of a final residual block after the attention-enhanced features are processed, followed by global average pooling and a fully connected layer to output the class logits. 

<img width="961" height="446" alt="image" src="https://github.com/user-attachments/assets/fe7b540f-d582-4011-bc29-cf517cf06f3e" />

## Dataset
Most of the dataset was custom-made, and it consisted of 2 parts: 
- 100 volunteers filled out a form with every letter of the Greek language in uppercase and lowercase
- Single characters were extracted from the ICDAR2012 Writer challenge dataset, which was provided by the National Centre for Scientific Research Demokritos

## Spelling Disorder
Leverages a Trie for dictionary spell checks and a word suggestion system based on Levenshtein distances. 

## Writing Disorder
Utilizes a 3-step algorithm to detect key patterns commonly found in passages written by individuals diagnosed with writing disorder.
- Word alignment irregularities
- Inconsistent capital letter usage
- Irregularity in spaces between words

<img width="776" height="189" alt="image" src="https://github.com/user-attachments/assets/9422df97-0fea-4f0e-aa5b-1dafccf5d5c5" />
<img width="772" height="160" alt="image" src="https://github.com/user-attachments/assets/57bf0eb9-efc8-454a-aa38-e8162a0208f8" />
