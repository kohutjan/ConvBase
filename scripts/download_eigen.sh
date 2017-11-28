rm -f ../Eigen.tar.gz
wget https://www.dl.dropboxusercontent.com/s/hhynuknplh3gwgp/Eigen.tar.gz -P ../include/
rm -rf ../Eigen
tar xzf ../include/Eigen.tar.gz -C ../include/
rm -f ../include/Eigen.tar.gz

