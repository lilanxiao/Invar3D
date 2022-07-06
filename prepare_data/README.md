# Preparation for ScanNet data

The code in this folder is modified from ScanNet [official repo](https://github.com/ScanNet/ScanNet). 

**Warning**: Due to the large amount of data, the code for data preparation is not fully tested yet. 

- First download ScanNet data from the [official repo](https://github.com/ScanNet/ScanNet). Note you need ~2TB space to save the raw data. 
- For efficient data extraction, you have compile the data reader. It is much faster than the Python implementation. 

        bash build.sh

- Before extracting the raw data, you need to update the paths in `unpack_img.py`. Note the unpacked data would further take hundreds of GB. Then run:
    
        python unpack_img.py
  
  It extracts depth maps, camera pose and RGB images. Also, RGB images are directly resized to the same size as depth maps to save storage space and computation during the training. This would take hours. 

- The extracted data should look like this:

        data_folder
        |
        |__ scene0000_00
        |   |
        |   |__ _info.txt               
        |   |__ frame-000000.color.jpg  
        |   |__ frame-000000.png        
        |   |__ frame-000000.pose.txt   
        |   |__ frame-000001.color.jpg
        |   |__ frame-000001.png
        |   |__ frame-000001.pose.txt
        |   |   ... ...
        |
        |__ scene0000_01
        |__ scene0000_02
        |   ... ...
    
    `frame-ID.png`: depth maps with 640x480. 

    `frame-ID.color.jpg`: color images with 640x480. 

    `frame-ID.pos.txt`: camera extrinsic. Not unsed in our training. 

    `_info.txt`: some meta data of the scene. Only the camera intrinsic in it is used. 
    
