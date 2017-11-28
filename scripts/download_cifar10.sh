mkdir -p ../data
rm -f ../data/cifar-10-batches-bin.tar.gz
wget https://dl.dropboxusercontent.com/s/auhkucn5d238qjk/cifar-10-batches-bin.tar.gz -P ../data/
rm -rf ../data/cifar-10-batches-bin
tar xzf ../data/cifar-10-batches-bin.tar.gz -C ../data/
rm -f ../data/cifar-10-batches-bin.tar.gz
