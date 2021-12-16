cd data
wget http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_background_images.zip
mkdir mnist_back_image
unzip mnist_background_images.zip -d mnist_back_image/
cd mnist_back_image 
mv mnist_background_images_test.amat test.amat
mv mnist_background_images_train.amat train.amat
cd ..


wget http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_back_image_new.zip
mkdir mnist_rot_back_image
unzip mnist_rotation_back_image_new.zip -d mnist_rot_back_image/
cd mnist_rot_back_image
mv mnist_all_background_images_rotation_normalized_test.amat test.amat
mv mnist_all_background_images_rotation_normalized_train_valid.amat train.amat
cd ..

wget http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip
mkdir mnist_rot
unzip mnist_rotation_new.zip -d mnist_rot/
cd mnist_rot
mv mnist_all_rotation_normalized_float_test.amat test.amat
mv mnist_all_rotation_normalized_float_train_valid.amat train.amat
cd ..

wget http://www.iro.umontreal.ca/~lisa/icml2007data/rectangles_images.zip
mkdir rectangles_image
unzip rectangles_images.zip -d rectangles_image/
cd rectangles_image
mv rectangles_im_train.amat train.amat
mv rectangles_im_test.amat test.amat
cd ..

wget http://www.iro.umontreal.ca/~lisa/icml2007data/rectangles.zip
mkdir rectangles
unzip rectangles.zip -d rectangles/
cd rectangles
mv rectangles_train.amat train.amat
mv rectangles_test.amat test.amat
cd ..

wget http://www.iro.umontreal.ca/~lisa/icml2007data/convex.zip
mkdir convex
unzip convex.zip -d convex/
cd convex
mv 50k/convex_test.amat .
mv convex_test.amat test.amat
mv convex_train.amat train.amat
cd ..

cd ..
