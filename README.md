A deep learning OCR model trained on greek letter recognition.
The model will take written text as an input and then recognize 
if the person who wrote it has dysorthographia or/and dysgraphia.

Handwritten Greek text OCR model: 
The OCR model is trained on data acquired from ~100 people who
signed and allowed us to use their data for this project. The
data consists of every possible single character of the greek 
language, as well as some double characters that often get written
in one stroke - so as to improve the model's accuracy. Finally,
both uppercase and lowercase letters are included, making the 
amount of classes equal to ~95.
The model is also trained to a class for space, so as to allow
us to know where to split each word invididually.
Data augmentation is also being used to slightly rotate images
and help us generate more data, making the amount for each
letter equal to ~300.
