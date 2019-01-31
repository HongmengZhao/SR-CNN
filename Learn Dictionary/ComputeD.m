function [Dictionary_text,Dictionary_picture]= ComputeD (I,I1,I2,I3,I4,I5,I6)  
%% The original SCI
 [img01,img02]=gradient(I);
 imgrad=sqrt(img01.^2+img02.^2);
%% The textual patches of SCI
[img11,img12]=gradient(I1);
imgrad1=sqrt(img11.^2+img12.^2);
[img21,img22]=gradient(I2);
imgrad2=sqrt(img21.^2+img22.^2);
[img31,img32]=gradient(I3);
imgrad3=sqrt(img31.^2+img32.^2);
%% The pictorial patches of SCI
[img41,img42]=gradient(I4);
imgrad4=sqrt(img41.^2+img42.^2);
[img51,img52]=gradient(I5);
imgrad5=sqrt(img51.^2+img52.^2);
[img61,img62]=gradient(I6);
imgrad6=sqrt(img61.^2+img62.^2);

numIterOfKsvd = 100;
K=100;
% [NN1,NN2] = size(imgrad);
[nn1,nn2] = size(imgrad1);
sigma = 5;
C = 1.15;
maxBlocksToConsider = 260000;
slidingDis = 4;
bb = 8;
maxNumBlocksToTrainOn = 20000;

if(prod([nn1,nn2]-bb+1)> ceil(maxNumBlocksToTrainOn/3))
    randPermutation =  randperm(prod([nn1,nn2]-bb+1));
    selectedBlocks = randPermutation(1:ceil(maxNumBlocksToTrainOn/3));
    %% The textual patch
    blkMatrix1 = zeros(bb^2,ceil(maxNumBlocksToTrainOn/3));
    blkMatrix2 = zeros(bb^2,ceil(maxNumBlocksToTrainOn/3));
    blkMatrix3 = zeros(bb^2,ceil(maxNumBlocksToTrainOn/3));
    %% The pictorial patch
    blkMatrix4 = zeros(bb^2,ceil(maxNumBlocksToTrainOn/3));
    blkMatrix5 = zeros(bb^2,ceil(maxNumBlocksToTrainOn/3));
    blkMatrix6 = zeros(bb^2,ceil(maxNumBlocksToTrainOn/3));
    
    for i = 1:ceil(maxNumBlocksToTrainOn/3)
        [row1,col1] = ind2sub(size(imgrad1)-bb+1,selectedBlocks(i));
        currBlock1 = imgrad1(row1:row1+bb-1,col1:col1+bb-1);
        blkMatrix1(:,i) = currBlock1(:);
        [row2,col2] = ind2sub(size(imgrad2)-bb+1,selectedBlocks(i));
        currBlock2 = imgrad2(row2:row2+bb-1,col2:col2+bb-1);
        blkMatrix2(:,i) = currBlock2(:);
        [row3,col3] = ind2sub(size(imgrad3)-bb+1,selectedBlocks(i));
        currBlock3 = imgrad3(row3:row3+bb-1,col3:col3+bb-1);
        blkMatrix3(:,i) = currBlock3(:);
        
        [row4,col4] = ind2sub(size(imgrad4)-bb+1,selectedBlocks(i));
        currBlock4 = imgrad4(row4:row4+bb-1,col4:col4+bb-1);
        blkMatrix4(:,i) = currBlock4(:);
        [row5,col5] = ind2sub(size(imgrad5)-bb+1,selectedBlocks(i));
        currBlock5 = imgrad5(row5:row5+bb-1,col5:col5+bb-1);
        blkMatrix5(:,i) = currBlock5(:);
        [row6,col6] = ind2sub(size(imgrad6)-bb+1,selectedBlocks(i));
        currBlock6 = imgrad6(row6:row6+bb-1,col6:col6+bb-1);
        blkMatrix6(:,i) = currBlock6(:);
    end
else
    blkMatrix1 = im2col(imgrad1,[bb,bb],'sliding');
    blkMatrix2 = im2col(imgrad2,[bb,bb],'sliding');
    blkMatrix3 = im2col(imgrad3,[bb,bb],'sliding');
    blkMatrix4 = im2col(imgrad4,[bb,bb],'sliding');
    blkMatrix5 = im2col(imgrad5,[bb,bb],'sliding');
    blkMatrix6 = im2col(imgrad6,[bb,bb],'sliding');
    
end
blkMatrix=zeros(bb^2,ceil(maxNumBlocksToTrainOn/3)*3);% belongs to text
BlkMatrix=zeros(bb^2,ceil(maxNumBlocksToTrainOn/3)*3);% belongs to picture

blkMatrix(:,1:ceil(maxNumBlocksToTrainOn/3))=blkMatrix1;
blkMatrix(:,ceil(maxNumBlocksToTrainOn/3)+1:ceil(maxNumBlocksToTrainOn/3)*2)=blkMatrix2;
blkMatrix(:,ceil(maxNumBlocksToTrainOn/3)*2+1:ceil(maxNumBlocksToTrainOn/3)*3)=blkMatrix3;

BlkMatrix(:,1:ceil(maxNumBlocksToTrainOn/3))=blkMatrix4;
BlkMatrix(:,ceil(maxNumBlocksToTrainOn/3)+1:ceil(maxNumBlocksToTrainOn/3)*2)=blkMatrix5;
BlkMatrix(:,ceil(maxNumBlocksToTrainOn/3)*2+1:ceil(maxNumBlocksToTrainOn/3)*3)=blkMatrix6;


param.K = K;
param.numIteration = numIterOfKsvd ;

param.errorFlag = 1; % decompose signals until a certain error is reached. do not use fix number of coefficients.
param.errorGoal = sigma*C;
param.preserveDCAtom = 0;

Pn=ceil(sqrt(K));
DCT=zeros(bb,Pn);
for k=0:1:Pn-1,
    V=cos([0:1:bb-1]'*k*pi/Pn);
    if k>0, V=V-mean(V); end;
    DCT(:,k+1)=V/norm(V);
end;
DCT=kron(DCT,DCT);

param.initialDictionary = DCT(:,1:param.K );
param.InitializationMethod =  'GivenMatrix';

% [Dictionary,output] = KSVD(blkMatrix,param);

[Dictionary_picture,~] = KSVD(BlkMatrix,param);
[Dictionary_text,~] = KSVD(blkMatrix,param);

Dictionary=zeros(bb^2,K*2);
Dictionary(:,1:K)=Dictionary_text;
Dictionary(:,K+1:K*2)=Dictionary_picture;

while (prod(floor((size(imgrad)-bb)/slidingDis)+1)>maxBlocksToConsider)
    slidingDis = slidingDis+1;
end
[blocks,idx] = my_im2col(im2double(imgrad),[bb,bb],slidingDis);

 

% go with jumps of 100000
for jj = 1:100000:size(blocks,2)
    jumpSize = min(jj+100000-1,size(blocks,2));
    vecOfMeans = mean(blocks(:,jj:jumpSize));
    blocks(:,jj:jumpSize) = blocks(:,jj:jumpSize) - repmat(vecOfMeans,size(blocks,1),1);
    GCoefs = OMPerr(Dictionary,blocks(:,jj:jumpSize),5*1.15);
    blocks(:,jj:jumpSize)= Dictionary*GCoefs + ones(size(blocks,1),1) * vecOfMeans;
end
end