# Fashion descriptions

get the dataset from:
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small?resource=download
extract it to kaggle-small folder on the root folder of the project

## Steps to go
 - generate descriptions using `generate-descriptions-blip.py`
 - refine descriptions `refine-descriptions.py` - it combines original description with generated description doing some preprocessing, e.g. eliminating generated words like 'a woman wears' 
 - `prepare_category_dataset.py` - saves the category mapping in the separate file
 - `prepare_color_dataset.py.py` - saves the color mapping in the separate file
 - `train_category_classifier.py` - train the model that predicts the product categorie based on description
 - `train_color_classifier.py` - train the model that predicts the product color based on description
 - `test_category_classifier.py` - brings a categorie based on product description
 - `test_color_classifier.py.py` - brings a color based on product description