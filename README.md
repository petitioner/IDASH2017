# IDASH2017

IDASH2017 is a project for implementing our Logistic Regression Traning on  encrypted datasets (Privacy-Preserving Logistic Regression Training with a Faster Gradient Variant )

The Python source codes to conduct the experiments in the clear and its results can be found in the folder "data" at IDASH2017/IDASH2017/data/. To run it, you need the Python version 2.7, NumPy (version 1.10.2) and matplotlib. 

## How to run this program? 

### Dependencies

On a Ubuntu cloud, our implementation requires the following libraries in order:
* `g++`:      
```sh
               # apt install g++ 
```

* `make`:       
```sh
                # apt install make
```

* `m4`: #        
```sh
                 # apt install m4
```

* `GMP`(ver. 6.1.2):      
```sh
                           # cd gmp-x.x.x  
                           # ./configure --enable-cxx  
                           # make
                           # make install
                           # ldconfig
```

* `NTL`(ver. 11.3.0): 
```sh
                     # cd ntl-x.x.x
                     # cd src
                     # ./configure NTL_THREADS=on NTL_THREAD_BOOST=on NTL_EXCEPTIONS=on
                     # make
                     # make install
```

### Running IDASH2017

You need to configure and build the CNNinference project. 

After that, in the 'Debug' folder, you can run our project by the following command lines:

```sh
# make clean
# make all
# ./MyIDASH2017
``` 

You can change the source codes and then repeat the above lines to debug your own project.

## Running a test source code

In the 'Debug' folder, you can find two running results:   

        'CNNinferArchiveFile20220407.7z_SetNumThreads(42)_nohup.out'  
        
        'CNNinferArchiveFile20220409.7z_SetNumThreads(42)_nohup.out'
        
        
Also, you can find the CSV file storing the weights of our well-trained CNN inference model at:

         HE.CNNinfer/CNNinference/data/CNNweightsMNIST.csv
         


            
            
    

