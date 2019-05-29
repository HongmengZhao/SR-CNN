function GCoefsame_ssim=FeatureComputation_PatchSelection(img0,img1,Dictionary_text,Dictionary_picture,center)
% Features Computation About structural similarity and Dictioanry Atom uasge overlapping rate.
% Input:  img0 -- The Reference SCI
%         img1 -- The Corresponding Distorte Version
%         Dictionary_text -- The learned textual dictionary of Reference SCI by KSVD
%         Dictionary_picture -- The learned pictorial dictionary of Reference SCI by KSVD
%         center -- The clustering result matrix
% Output: GCoefsame_ssim -- The Adaptively Selective patches of The Dictioanry Atom Uasge Overlapping Rate And Structural Similarity Patch
% between the reference and distorted SCI

[img01,img02]=gradient(img0);
imgrad=sqrt(img01.^2+img02.^2);
[NN1,NN2] = size(imgrad);

[img11,img12]=gradient(img1);
imgrad1=sqrt(img11.^2+img12.^2);

%% The structual similarity computation
[~ ,ssim_map]=ssim_index(imgrad,imgrad1);

%% The parameters of image reconstructure
maxBlocksToConsider = 2600000;
sigma =5;
C = 1.15;%errorgoal
slidingDis =4;
bb = 8;

K=size(Dictionary_text,2);
Dictionary=zeros(bb^2,K*2);
Dictionary(:,1:K)=Dictionary_text;
Dictionary(:,K+1:K*2)=Dictionary_picture;

while (prod(floor((size(imgrad)-bb)/slidingDis)+1)>maxBlocksToConsider)
    slidingDis = slidingDis+1;
end
[blocks,idx] = my_im2col(im2double(imgrad),[bb,bb],slidingDis);

% go with jumps of 40000
for jj = 1:100000:size(blocks,2)
    jumpSize = min(jj+100000-1,size(blocks,2));
    vecOfMeans = mean(blocks(:,jj:jumpSize));
    blocks(:,jj:jumpSize) = blocks(:,jj:jumpSize) - repmat(vecOfMeans,size(blocks,1),1);
    GCoefs = OMPerr(Dictionary,blocks(:,jj:jumpSize),sigma*C);
    blocks(:,jj:jumpSize)= Dictionary*GCoefs + ones(size(blocks,1),1) * vecOfMeans;
end

while (prod(floor((size(imgrad1)-bb)/slidingDis)+1)>maxBlocksToConsider)
    slidingDis = slidingDis+1;
end
[blocks1,idx] = my_im2col(im2double(imgrad1),[bb,bb],slidingDis);


for jj = 1:100000:size(blocks1,2)
    jumpSize1 = min(jj+100000-1,size(blocks1,2));
    vecOfMeans1 = mean(blocks1(:,jj:jumpSize1));
    blocks1(:,jj:jumpSize1) = blocks1(:,jj:jumpSize1) - repmat(vecOfMeans1,size(blocks1,1),1);
    GCoefs2 = OMPerr(Dictionary,blocks1(:,jj:jumpSize1),sigma*C);
end

%% The Dictionary consistency degree analysis
        GCoefsame=[];%Dictionary atom usage overlapping
        GCoefweight=[];%Dictionary atom usage coefficient
        for i=1:size(GCoefs,2)
            if numel(find(GCoefs(:,i)~=0))~=0
                GCoefweight(1,i)=((abs(sum(GCoefs2(find(GCoefs2(:,i)~=0),i)))-sum(GCoefs(find(GCoefs(:,i)~=0,i))))./(sum(abs(GCoefs2(find(GCoefs2(:,i)~=0),i)))+sum(abs(GCoefs(find(GCoefs(:,i)~=0),i))))).^0.6;
                GCoefsame(1,i)=numel(intersect(find(GCoefs2(:,i)~=0),find(GCoefs(:,i)~=0)))./(numel(find(GCoefs(:,i)~=0))+numel(find(GCoefs2(:,i)~=0)));
            else
                GCoefsame(1,i)=0;
                GCoefweight(1,i)=0;
            end
        end
         sumGcoef=[];
         sumGcoef=GCoefsame.*(GCoefweight.^0.2);

%% Adaptively Patch Selection
GCoefsame_ssim=zeros(224,224,2,size(center,1));
for i=1:size(center,1)
    x=center(i,1);y=center(i,2);
    rows=ceil((NN1-bb+slidingDis)/slidingDis);cols=ceil((NN2-bb+slidingDis)/slidingDis);
    row_x1=ceil((x-bb+slidingDis)/slidingDis);col_y1=ceil((y-bb+slidingDis)/slidingDis);
    if (row_x1<1);row_x1=1; end
    if (col_y1<1);col_y1=1; end
    for rows_i=1:57
        GCoefnumsame(1,(rows_i-1)*57+1:rows_i*57)=sumGcoef((row_x1+rows_i-2)*rows+col_y1:(row_x1+rows_i-2)*rows+col_y1+56);
    end
    ssim=imcrop(ssim_map,[x,y,223,223]);% Structural Similarity Patch
    GCoefsame=imresize(reshape(GCoefnumsame,57,57), [224 224]);%The Dictioanry Atom Uasge Overlapping Rate Patch 
    GCoefsame_ssim(:,:,1,i)=ssim;
    GCoefsame_ssim(:,:,2,i)=GCoefsame;  
end
end