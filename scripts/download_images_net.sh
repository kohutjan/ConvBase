mkdir -p ../data
rm -f ../data/random_class_images.tar.gz
wget https://dl.dropboxusercontent.com/s/ccajtz0jgu6h5pp/random_class_images.tar.gz -P ../data/
rm -rf ../data/random_class_images
tar xzf ../data/random_class_images.tar.gz -C ../data/
rm -f ../data/random_class_images.tar.gz
mkdir -p ../nets
rm -f ../nets/2xC32_P2_2xC32_P2_2xC32_P2_FC256_FC10_iter_17500.convbase
wget https://dl.dropboxusercontent.com/s/w7q89m8a76b6846/2xC32_P2_2xC32_P2_2xC32_P2_FC256_FC10_iter_17500.convbase -P ../nets/

