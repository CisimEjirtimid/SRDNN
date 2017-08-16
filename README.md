# dlib_SRDNN

## Super-Resolution using Deep Neural Networks, a *C++ Project*

This application was made with usage of **dlib** computer vision library, and **argagg** argument parsing library.

dlib version - 19.4
argagg version - 0.46

## Example usage

``` Training
-t -i "F:\migration backup\SRDNN\VOC2012\500x375 - 700" -o "simple_network.dat" -x "simple_network.xml"
```

``` Evaluation
-e -i "F:\migration backup\SRDNN\VOC2012\500x375 - 7000\2010_001002.jpg" -n "simple_network.dat" -s
```

For more detailed usage, try `--help`