echo "Removing old folders..."
rm -rf foot_cfg*
echo "Removing old models..."
cd models
rm -rf photos_src
rm -rf test1
cd ..
echo "Removing old photos..."
cd photos
cd FAMEST
rm -rf photos_src
cd ..
cd ..
echo "Removing old videos..."
cd videos
cd FAMEST
rm -rf test1
cd ..
cd ..
echo "Rebuilding docker..."
cd ..
docker-compose up --build --force-recreate -d
echo "Running docker..."
#cd ..
docker-compose exec python /bin/sh
