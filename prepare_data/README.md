# Preperation for ScanNet data

The code in this folder is modified from ScanNet [official repo](https://github.com/ScanNet/ScanNet). 
**Warning**: Due to the large amount of data, the code for data preparation is not fully tested yet. 

- First download ScanNet data from the [official repo](https://github.com/ScanNet/ScanNet). Note you need ~2TB space to save the raw data. 
- For efficient data extraction, you have compile the data reader. It is much faster than the Python implementation. 

        bash build.sh

- Then extract the raw data, you need to first update the paths in `unpack_img.py`. Note the unpacked data would further take hunderds of GB. Then run:
    
        python unpack_img.py
  
  It extracts depth maps, camera pose and RGB images. Also, RGB images are resize to the same size as depth maps. This would take hours. 
