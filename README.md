# Time series classification based on Unet with 2D transformation

** This work has been accepted by IIKI2022.  The link for our paper will be provided after the official publication** 

Recently, time series classification has attracted great interest in the data mining community. Hundreds of time series classification methods have been proposed in recent decades. However, these methods only extract features from the one-dimensional space, leading to relatively low classification accuracy. For this, a Time Series Classification method based on Unet with 2D transformation (TSC-Unet) is proposed in this paper. This method first transforms the univariate time series into 2D space and augments them. After that, features are extracted from 2D space through a fully convolutional network. Finally,  TSC-Unet is trained to achieve high accuracy. Experimental results on the UCR datasets demonstrate the proposed method has high classification accuracy.

## Acknowledgements
This work was supported by the Innovation Methods Work Special Project under Grant 2020IM020100, and the Natural Science Foundation of Shandong Province under Grant ZR2020QF112.

We would like to thank Eamonn Keogh and his team, Tony Bagnall and his team for the UEA/UCR time series classification repository.
