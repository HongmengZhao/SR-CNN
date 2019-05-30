clc;
clear;
%% the reference and corresponding distorted version and text segmentaion index map
I_ref = double(rgb2gray(imread('.\img\cim13.bmp')));
I_dis = double(rgb2gray(imread('.\img\cim13_3_4.bmp')));
I_index = double(imread('.\img\cim13_segIndex.bmp'));
load '.\Dictionary\Dictionary.mat';
%% Obtain the clustering centers
center = kmeans_HOG(I_ref,I_index,12);

%% Features Computation & Adaptively Patch Selection
GCoefsame_ssim = Patchwise_FeatureComputation(I_ref,I_dis,Dictionary_text,Dictionary_picture,center);

%% Get the overall score by trained model
test_data=zeros(224,224,2,12,'single');
test_data=GCoefsame_ssim;
score=test(test_data);