# Fashion descriptions

get the dataset from:
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small?resource=download
extract it to kaggle-small folder on the root folder of the project

## Steps to go
 - generate descriptions using `generate-descriptions-blip.py`, it uses blip model to generate additional descriptions based on image representation
 - refine descriptions `refine-descriptions.py` - it combines original description with generated description doing some preprocessing, e.g. eliminating generated words like 'a woman wears' 
 - `prepare_category_dataset.py` - saves the category mapping in the separate file
 - `prepare_color_dataset.py` - saves the color mapping in the separate file
 - `train_category_classifier.py` - train the model that predicts the product categorie based on description
 - `train_color_classifier.py` - train the model that predicts the product color based on description
 - `test_category_classifier.py` - brings a categorie based on product description
 - `test_color_classifier.py` - brings a color based on product description
 - `train_vit.py` - train vit model to predict color or category based on product image
   - change the TARGET_COL variable value to refer either to color or category attribute to train
 - `test_vit.py` - test vit model which predicts color or category based on product image
    - change the TARGET variable value to refer either to color or category attribute
    - change image_path to refer to different target images to test

now it's getting into practical part we generate embeddings combining the product descriptions 
and product images. We will use both models (for category and color) to generate embeddings for the products
 - run `extract_vit_embeddings.py` for getting embeddings based on images
   - change TARGET variable to get embeddings respectively for color and category
 - run `extract_text_embeddings.py` for getting embeddings based on text refined by `refine-descriptions.py`

[//]: # ( - run `combine_vit_embeddings.py` to combine both articleType and baseColor vit embeddings)

[//]: # ()
[//]: # (We got couple of embeddings files: `embeddings_text.npy` and `embeddings_combined.npy` but those are of different shape and )

[//]: # (cosine similarity requires both vectors to have the same dimensionality, why?)

[//]: # ( - DistilBERT text model outputs 384-dimensional embeddings)

[//]: # ( - ViT combined model outputs 1536-dimensional embeddings &#40;likely 768 per model × 2 concatenated&#41;)

[//]: # ( - So they “live” in different spaces — not directly comparable.)

[//]: # ()
[//]: # (We'd need to project both into a shared dimension)

[//]: # ( - run `create_embeddings_projection.py`)

Run `get_similar_by_text.py` and change the query string to see a result

