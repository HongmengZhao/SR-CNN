clc;
clear;
%% the reference and corresponding distorted version and text segmentaion index map
I_ref = double(rgb2gray(imread('cim13.bmp')));
I_dis = double(rgb2gray(imread('cim13_3_4.bmp')));
I_index = double(imread('cim13_segIndex.bmp'));
load 'Dictionary.mat';
%% Obtain the clustering centers
center = HOG_kmeans(I_ref,I_index);

%% Features Computation & Adaptively Patch Selection
GCoefsame_ssim = FeatureComputation_PatchSelection(I_ref,I_dis,Dictionary_text,Dictionary_picture,center);

%% Get the overall score by trained model
test_data=zeros(224,224,2,6,'single');
test_data=GCoefsame_ssim;
score=test(test_data);