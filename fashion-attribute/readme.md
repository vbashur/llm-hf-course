# Fashion descriptions

get the dataset from:
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small?resource=download
extract it to kaggle-small folder on the root folder of the project

## Steps to go
 - generate descriptions using `generate-descriptions-blip.py`
 - refine descriptions `refine-descriptions.py` - it combines original description with generated description doing some preprocessing, e.g. eliminating generated words like 'a woman wears' 
 - `prepare_datasets.py` - saves the category mapping in the separate file
 - `train_classifier.py` - train the model that predicts the product categorie based on description
 - `test_classifier.py` - brings a categorie based on product desctiption