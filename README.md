An AI OCR model trained on greek letter recognition.
The final code will take written text as an input and then recognize 
if the person who wrote it has dysorthographia or/and dysgraphia.

Handwritten Greek text OCR model: 
The OCR model is trained on data acquired from ~100 people who
signed and allowed us to use their data for this project. The
data consists of every possible single character of the greek 
language, as well as some double characters that often get written
in one stroke - so as to improve the model's accuracy. Finally,
both uppercase and lowercase letters are included, making the 
amount of classes equal to ~95.
Data augmentation is also being used to slightly rotate images
and help us generate more data, creating ~5000 new augmentations
for each image. We also utilize different kinds of augmentations 
while the model is training, so as to make it learn patterns
better.
